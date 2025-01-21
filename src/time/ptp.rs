use std::fs::File;
use std::time::{Duration, SystemTime};

#[cfg(unix)]
use nix::{
    sys::ioctl,
    sys::{
        socket::{socket, AddressFamily, SockFlag, SockType, SockProtocol},
        time::TimeSpec,
    },
};
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
#[cfg(unix)]
use std::io::Write;

use crate::core::{Error, Result};
use super::source::{HardwareTimestamp, TimeSource};

// PTP message types
const SYNC: u8 = 0x00;
const DELAY_REQ: u8 = 0x01;
const FOLLOW_UP: u8 = 0x08;
const DELAY_RESP: u8 = 0x09;

// PTP header flags
const TWO_STEP: u16 = 0x0200;

/// PTP hardware clock device
pub struct PtpClock {
    /// Device file handle
    device: File,
    /// Socket for PTP messages
    #[cfg(unix)]
    socket: RawFd,
    #[cfg(not(unix))]
    socket: i32,
    /// Domain number
    domain: u8,
    /// Current sequence ID
    sequence_id: u16,
    /// Last sync timestamp
    last_sync: Option<HardwareTimestamp>,
    /// Last follow-up timestamp
    last_followup: Option<Duration>,
    /// Last delay request timestamp
    last_delay_req: Option<HardwareTimestamp>,
    /// Last delay response timestamp
    last_delay_resp: Option<Duration>,
}

impl PtpClock {
    /// Opens a PTP hardware clock device
    #[cfg(unix)]
    pub fn open(device_path: &str, domain: u8) -> Result<Self> {
        // Open PTP device
        let device = File::open(device_path)
            .map_err(|e| Error::timing(format!("Failed to open PTP device: {}", e)))?;

        // Create PTP socket
        let socket = socket(
            AddressFamily::Inet,
            SockType::Raw,
            SockFlag::empty(),
            Some(SockProtocol::Udp),
        ).map_err(|e| Error::timing(format!("Failed to create PTP socket: {}", e)))?;

        // Hardware timestamp configuration
        #[derive(Copy, Clone)]
        #[repr(C)]
        struct HwTimestampConfig {
            flags: i32,
            tx_type: i32,
            rx_filter: i32,
        }

        const HWTSTAMP_TX_ON: i32 = 1;
        const HWTSTAMP_FILTER_PTP_V2_EVENT: i32 = 1;
        const SIOCSHWTSTAMP: u64 = 0x89b0;

        let mut ts_config = HwTimestampConfig {
            flags: 0,
            tx_type: HWTSTAMP_TX_ON,
            rx_filter: HWTSTAMP_FILTER_PTP_V2_EVENT,
        };

        // Configure hardware timestamping
        unsafe {
            if ioctl::ioctl(device.as_raw_fd(), SIOCSHWTSTAMP, &mut ts_config as *mut _) < 0 {
                return Err(Error::timing("Failed to configure hardware timestamping"));
            }
        }

        Ok(PtpClock {
            device,
            socket,
            domain,
            sequence_id: 0,
            last_sync: None,
            last_followup: None,
            last_delay_req: None,
            last_delay_resp: None,
        })
    }

    #[cfg(not(unix))]
    pub fn open(_device_path: &str, domain: u8) -> Result<Self> {
        Err(Error::timing("PTP hardware clock not supported on this platform"))
    }

    /// Captures a hardware timestamp
    #[cfg(unix)]
    fn capture_timestamp(&self) -> Result<HardwareTimestamp> {
        // Hardware timestamp data
        #[derive(Copy, Clone)]
        #[repr(C)]
        struct HwTimestamp {
            sec: i64,
            nsec: i32,
            flags: u32,
        }

        const SIOCGHWTSTAMP: u64 = 0x89b1;

        let mut ts = HwTimestamp {
            sec: 0,
            nsec: 0,
            flags: 0,
        };

        unsafe {
            if ioctl::ioctl(self.device.as_raw_fd(), SIOCGHWTSTAMP, &mut ts as *mut _) < 0 {
                return Err(Error::timing("Failed to read hardware timestamp"));
            }
        }

        Ok(HardwareTimestamp {
            software: SystemTime::now(),
            hardware: ts.sec as u64 * 1_000_000_000 + ts.nsec as u64,
            source: TimeSource::Ptp(self.device.as_raw_fd().to_string()),
            error_bound: 100, // 100ns error bound for hardware timestamps
        })
    }

    #[cfg(not(unix))]
    fn capture_timestamp(&self) -> Result<HardwareTimestamp> {
        Ok(HardwareTimestamp {
            software: SystemTime::now(),
            hardware: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_nanos() as u64,
            source: TimeSource::Ptp("software".to_string()),
            error_bound: 1_000_000, // 1ms error bound for software timestamps
        })
    }

    /// Sends a PTP sync message
    #[cfg(unix)]
    pub fn send_sync(&mut self) -> Result<HardwareTimestamp> {
        let mut msg = [0u8; 44]; // Basic sync message size
        
        // PTP header
        msg[0] = SYNC;
        msg[1] = 2; // PTP version 2
        msg[2..4].copy_from_slice(&(34u16).to_be_bytes()); // Message length
        msg[4] = self.domain;
        msg[6..8].copy_from_slice(&self.sequence_id.to_be_bytes());
        msg[8] = 0x00; // Control field (Sync)
        msg[9] = 0x02; // Log mean message interval

        self.sequence_id = self.sequence_id.wrapping_add(1);

        // Send message
        #[cfg(unix)]
        {
            let mut file = unsafe { std::fs::File::from_raw_fd(self.socket) };
            file.write_all(&msg)
                .map_err(|e| Error::timing(format!("Failed to send sync message: {}", e)))?;
            std::mem::forget(file); // Don't close the socket
        }
    #[cfg(not(unix))]
        return Err(Error::timing("PTP not supported on this platform"));

        // Capture transmit timestamp
        let ts = self.capture_timestamp()?;
        self.last_sync = Some(ts.clone());
        Ok(ts)
    }

    #[cfg(not(target_family = "unix"))]
    pub fn send_sync(&mut self) -> Result<HardwareTimestamp> {
        Err(Error::timing("PTP not supported on this platform"))
    }

    /// Sends a PTP follow-up message
    #[cfg(unix)]
    pub fn send_followup(&mut self, sync_time: Duration) -> Result<()> {
        let mut msg = [0u8; 44];
        
        // PTP header
        msg[0] = FOLLOW_UP;
        msg[1] = 2;
        msg[2..4].copy_from_slice(&(34u16).to_be_bytes());
        msg[4] = self.domain;
        msg[6..8].copy_from_slice(&(self.sequence_id - 1).to_be_bytes());
        msg[8] = 0x02; // Control field (Follow_Up)

        // Precise origin timestamp
        let secs = sync_time.as_secs();
        let nsecs = sync_time.subsec_nanos();
        msg[34..38].copy_from_slice(&(secs as u32).to_be_bytes());
        msg[38..42].copy_from_slice(&nsecs.to_be_bytes());

    #[cfg(unix)]
        {
            let mut file = unsafe { std::fs::File::from_raw_fd(self.socket) };
            file.write_all(&msg)
                .map_err(|e| Error::timing(format!("Failed to send follow-up message: {}", e)))?;
            std::mem::forget(file); // Don't close the socket
        }
    #[cfg(not(unix))]
        return Err(Error::timing("PTP not supported on this platform"));

        self.last_followup = Some(sync_time);
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn send_followup(&mut self, _sync_time: Duration) -> Result<()> {
        Err(Error::timing("PTP not supported on this platform"))
    }

    /// Handles an incoming PTP message
    pub fn handle_message(&mut self, msg: &[u8]) -> Result<Option<Duration>> {
        if msg.len() < 34 {
            return Ok(None);
        }

        match msg[0] & 0x0F {
            SYNC => {
                let ts = self.capture_timestamp()?;
                self.last_sync = Some(ts);
                Ok(None)
            }
            FOLLOW_UP => {
                if let Some(sync_ts) = &self.last_sync {
                    let secs = u32::from_be_bytes(msg[34..38].try_into().unwrap()) as u64;
                    let nsecs = u32::from_be_bytes(msg[38..42].try_into().unwrap());
                    let origin_time = Duration::new(secs, nsecs);
                    
                    // Calculate offset
                    let hw_time = Duration::from_nanos(sync_ts.hardware);
                    let offset = if hw_time > origin_time {
                        hw_time - origin_time
                    } else {
                        origin_time - hw_time
                    };
                    
                    Ok(Some(offset))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None)
        }
    }

    /// Gets the current PTP time
    pub fn get_time(&self) -> Result<SystemTime> {
        let now = SystemTime::now();
        let offset = self.last_followup.unwrap_or(Duration::from_secs(0));
        Ok(now + offset)
    }
}

impl Drop for PtpClock {
    #[cfg(target_family = "unix")]
    fn drop(&mut self) {
        #[cfg(target_family = "unix")]
        unsafe {
            let _ = std::fs::File::from_raw_fd(self.socket);
            // File is closed when dropped
        }
    }

    #[cfg(not(target_family = "unix"))]
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    #[ignore] // Requires PTP hardware
    fn test_ptp_timestamp() {
        if let Ok(mut clock) = PtpClock::open("/dev/ptp0", 0) {
            // Send sync message and capture timestamp
            if let Ok(ts) = clock.send_sync() {
                assert!(ts.hardware > 0);
                assert!(ts.error_bound <= 100);
            }
        }
    }

    #[test]
    #[ignore] // Requires PTP hardware
    fn test_ptp_offset() {
        if let Ok(mut clock) = PtpClock::open("/dev/ptp0", 0) {
            // Send sync and follow-up
            if let Ok(sync_ts) = clock.send_sync() {
                let sync_time = Duration::from_nanos(sync_ts.hardware);
                if let Ok(()) = clock.send_followup(sync_time) {
                    // Simulate receiving follow-up message
                    let mut msg = [0u8; 44];
                    msg[0] = FOLLOW_UP;
                    msg[1] = 2;
                    let secs = sync_time.as_secs();
                    let nsecs = sync_time.subsec_nanos();
                    msg[34..38].copy_from_slice(&(secs as u32).to_be_bytes());
                    msg[38..42].copy_from_slice(&nsecs.to_be_bytes());

                    if let Ok(Some(offset)) = clock.handle_message(&msg) {
                        // Offset should be small for local timestamps
                        assert!(offset < Duration::from_micros(100));
                    }
                }
            }
        }
    }
}
