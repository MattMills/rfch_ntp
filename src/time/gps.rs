use std::fs::File;
use std::io::{BufRead, BufReader};
#[cfg(target_family = "unix")]
use std::os::unix::io::AsRawFd;
use std::time::{Duration, SystemTime};
use serialport::SerialPort;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc, Datelike, Timelike};

use crate::core::{Error, Result};
use super::source::{HardwareTimestamp, TimeSource};

/// NMEA sentence types we care about
const GPRMC: &str = "$GPRMC";
const GPZDA: &str = "$GPZDA";

/// GPS receiver device
pub struct GpsReceiver {
    /// Serial port for GPS data
    port: Box<dyn SerialPort>,
    /// PPS device file
    #[cfg(target_family = "unix")]
    pps: Option<File>,
    /// Last received timestamp
    last_time: Option<SystemTime>,
    /// Last PPS event
    last_pps: Option<SystemTime>,
    /// Accumulated error estimate (nanoseconds)
    error_estimate: u64,
}

impl GpsReceiver {
    /// Opens a GPS receiver device
    pub fn open(device_path: &str, baud_rate: u32) -> Result<Self> {
        // Open serial port
        let port = serialport::new(device_path, baud_rate)
            .timeout(Duration::from_millis(100))
            .open()
            .map_err(|e| Error::timing(format!("Failed to open GPS device: {}", e)))?;

        // Try to open PPS device if available
        #[cfg(target_family = "unix")]
        let pps = File::open("/dev/pps0").ok();

        // Enable PPS if available
        #[cfg(target_family = "unix")]
        if let Some(ref pps) = pps {
            use nix::sys::ioctl;
            use nix::sys::time::TimeSpec;

            // PPS capture parameters
            #[derive(Copy, Clone)]
            #[repr(C)]
            struct PpsKParams {
                api_version: i32,
                mode: i32,
                assert_off_tu: TimeSpec,
                clear_off_tu: TimeSpec,
            }

            const PPS_CAPTUREASSERT: i32 = 0x01;
            const PPS_SETPARAMS: u64 = 0x40105001;

            let mut caps = PpsKParams {
                api_version: 1,
                mode: PPS_CAPTUREASSERT,
                assert_off_tu: TimeSpec::new(0, 0),
                clear_off_tu: TimeSpec::new(0, 0),
            };

            unsafe {
                if ioctl::ioctl(pps.as_raw_fd(), PPS_SETPARAMS, &mut caps as *mut _) < 0 {
                    return Err(Error::timing("Failed to configure PPS device"));
                }
            }
        }

        Ok(GpsReceiver {
            port,
            #[cfg(target_family = "unix")]
            pps,
            last_time: None,
            last_pps: None,
            error_estimate: 1_000_000, // Start with 1ms error estimate
        })
    }

    /// Reads and parses the next NMEA sentence
    fn read_sentence(&mut self) -> Result<Option<DateTime<Utc>>> {
        let mut reader = BufReader::new(&mut self.port);
        let mut line = String::new();
        reader.read_line(&mut line)?;

        // Parse NMEA sentence
        if line.starts_with(GPRMC) {
            self.parse_rmc(&line)
        } else if line.starts_with(GPZDA) {
            self.parse_zda(&line)
        } else {
            Ok(None)
        }
    }

    /// Parses RMC (Recommended Minimum) sentence
    fn parse_rmc(&self, sentence: &str) -> Result<Option<DateTime<Utc>>> {
        let fields: Vec<&str> = sentence.split(',').collect();
        if fields.len() < 10 {
            return Ok(None);
        }

        // Parse time and date fields
        let time = fields[1];
        let date = fields[9];
        if time.len() != 6 || date.len() != 6 {
            return Ok(None);
        }

        let hour = time[0..2].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;
        let minute = time[2..4].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;
        let second = time[4..6].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;

        let day = date[0..2].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;
        let month = date[2..4].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;
        let year = 2000 + date[4..6].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;

        let dt = NaiveDateTime::new(
            chrono::NaiveDate::from_ymd_opt(year as i32, month, day).unwrap(),
            chrono::NaiveTime::from_hms_opt(hour, minute, second).unwrap(),
        );

        Ok(Some(Utc.from_utc_datetime(&dt)))
    }

    /// Parses ZDA (Time & Date) sentence
    fn parse_zda(&self, sentence: &str) -> Result<Option<DateTime<Utc>>> {
        let fields: Vec<&str> = sentence.split(',').collect();
        if fields.len() < 7 {
            return Ok(None);
        }

        let time = fields[1];
        if time.len() != 6 {
            return Ok(None);
        }

        let hour = time[0..2].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;
        let minute = time[2..4].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;
        let second = time[4..6].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS time format"))?;

        let day = fields[2].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;
        let month = fields[3].parse::<u32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;
        let year = fields[4].parse::<i32>()
            .map_err(|_| Error::timing("Invalid GPS date format"))?;

        let dt = NaiveDateTime::new(
            chrono::NaiveDate::from_ymd_opt(year, month, day).unwrap(),
            chrono::NaiveTime::from_hms_opt(hour, minute, second).unwrap(),
        );

        Ok(Some(Utc.from_utc_datetime(&dt)))
    }

    /// Reads PPS timestamp if available
    #[cfg(target_family = "unix")]
    fn read_pps(&mut self) -> Result<Option<SystemTime>> {
        if let Some(ref pps) = self.pps {
            use nix::sys::ioctl;
            use nix::sys::time::TimeSpec;

            // PPS event data
            #[derive(Copy, Clone)]
            #[repr(C)]
            struct PpsInfo {
                assert_sequence: u32,
                clear_sequence: u32,
                assert_tu: TimeSpec,
                clear_tu: TimeSpec,
            }

            const PPS_GETTIME: u64 = 0x80105002;

            let mut event = PpsInfo {
                assert_sequence: 0,
                clear_sequence: 0,
                assert_tu: TimeSpec::new(0, 0),
                clear_tu: TimeSpec::new(0, 0),
            };

            unsafe {
                if ioctl::ioctl(pps.as_raw_fd(), PPS_GETTIME, &mut event as *mut _) >= 0 {
                    let ts = SystemTime::UNIX_EPOCH + Duration::new(
                        event.assert_tu.tv_sec() as u64,
                        event.assert_tu.tv_nsec() as u32,
                    );
                    self.last_pps = Some(ts);
                    return Ok(Some(ts));
                }
            }
        }
        Ok(None)
    }

    #[cfg(not(target_family = "unix"))]
    fn read_pps(&mut self) -> Result<Option<SystemTime>> {
        Ok(None)
    }

    /// Gets a timestamp with hardware precision if available
    pub fn get_timestamp(&mut self) -> Result<HardwareTimestamp> {
        // Read PPS first if available
        let pps_time = self.read_pps()?;
        
        // Read NMEA data
        let gps_time = self.read_sentence()?;

        // Calculate timestamp and error estimate
        let (timestamp, error) = match (pps_time, gps_time) {
            (Some(pps), Some(gps)) => {
                // Use PPS for precise timing
                let gps_systime = SystemTime::UNIX_EPOCH + 
                    Duration::from_secs(gps.timestamp() as u64);
                
                // Adjust error estimate based on PPS availability
                self.error_estimate = 100; // 100ns with PPS
                (pps, self.error_estimate)
            }
            (None, Some(gps)) => {
                // Use GPS time only
                let ts = SystemTime::UNIX_EPOCH + 
                    Duration::from_secs(gps.timestamp() as u64);
                self.error_estimate = 1_000_000; // 1ms without PPS
                (ts, self.error_estimate)
            }
            _ => {
                // Fall back to system time
                self.error_estimate = 10_000_000; // 10ms fallback
                (SystemTime::now(), self.error_estimate)
            }
        };

        self.last_time = Some(timestamp);

        Ok(HardwareTimestamp {
            software: SystemTime::now(),
            hardware: timestamp
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_nanos() as u64,
            source: TimeSource::Gps {
                device: "GPS".to_string(),
                use_pps: pps_time.is_some(),
            },
            error_bound: error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPS hardware
    fn test_rmc_parsing() {
        let port = serialport::new("/dev/null", 9600).open().unwrap();
        let receiver = GpsReceiver {
            port,
            #[cfg(target_family = "unix")]
            pps: None,
            last_time: None,
            last_pps: None,
            error_estimate: 0,
        };

        let sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A";
        let time = receiver.parse_rmc(sentence).unwrap().unwrap();
        
        assert_eq!(time.hour(), 12);
        assert_eq!(time.minute(), 35);
        assert_eq!(time.second(), 19);
        assert_eq!(time.day(), 23);
        assert_eq!(time.month(), 3);
        assert_eq!(time.year(), 1994);
    }

    #[test]
    #[ignore] // Requires GPS hardware
    fn test_zda_parsing() {
        let port = serialport::new("/dev/null", 9600).open().unwrap();
        let receiver = GpsReceiver {
            port,
            #[cfg(target_family = "unix")]
            pps: None,
            last_time: None,
            last_pps: None,
            error_estimate: 0,
        };

        let sentence = "$GPZDA,123519.00,23,03,1994,00,00*6A";
        let time = receiver.parse_zda(sentence).unwrap().unwrap();
        
        assert_eq!(time.hour(), 12);
        assert_eq!(time.minute(), 35);
        assert_eq!(time.second(), 19);
        assert_eq!(time.day(), 23);
        assert_eq!(time.month(), 3);
        assert_eq!(time.year(), 1994);
    }
}
