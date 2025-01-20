use bytes::{Buf, BufMut, Bytes, BytesMut};
use tokio_util::codec::{Decoder, Encoder};
use std::io;
use std::time::SystemTime;

use crate::core::{Error, FrequencyComponent, NodeId, Tier};
use super::message::{Message, TierInfo, FrequencyBand, TierChangeReason};

/// Protocol message codec for encoding/decoding network messages
#[derive(Clone, Default)]
pub struct MessageCodec;

impl MessageCodec {
    /// Creates a new message codec
    pub fn new() -> Self {
        MessageCodec
    }
}

impl Decoder for MessageCodec {
    type Item = Message;
    type Error = Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.len() < 4 {
            // Need more data to read message length
            return Ok(None);
        }

        // Read message length
        let mut length_bytes = [0u8; 4];
        length_bytes.copy_from_slice(&src[..4]);
        let length = u32::from_be_bytes(length_bytes) as usize;

        if src.len() < 4 + length {
            // Need more data to read full message
            return Ok(None);
        }

        // Remove length bytes
        src.advance(4);

        // Take message bytes
        let message_bytes = src.split_to(length);

        // Deserialize message
        match bincode::deserialize(&message_bytes) {
            Ok(message) => Ok(Some(message)),
            Err(e) => Err(Error::protocol(format!("Failed to deserialize message: {}", e))),
        }
    }
}

impl Encoder<Message> for MessageCodec {
    type Error = Error;

    fn encode(&mut self, item: Message, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // Serialize message
        let bytes = bincode::serialize(&item)
            .map_err(|e| Error::protocol(format!("Failed to serialize message: {}", e)))?;

        // Write length prefix
        dst.put_u32(bytes.len() as u32);

        // Write message bytes
        dst.extend_from_slice(&bytes);

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BytesMut;

    #[test]
    fn test_codec_hello_message() {
        let mut codec = MessageCodec::new();
        let mut bytes = BytesMut::new();

        let node_id = NodeId::random();
        let message = Message::Hello {
            node_id: node_id.clone(),
            tier: Tier::new(0),
            version: 1,
        };

        // Encode
        codec.encode(message.clone(), &mut bytes).unwrap();

        // Decode
        if let Some(decoded) = codec.decode(&mut bytes).unwrap() {
            match (message, decoded) {
                (Message::Hello { node_id: id1, tier: t1, version: v1 },
                 Message::Hello { node_id: id2, tier: t2, version: v2 }) => {
                    assert_eq!(id1, id2);
                    assert_eq!(t1, t2);
                    assert_eq!(v1, v2);
                }
                _ => panic!("Decoded wrong message type"),
            }
        } else {
            panic!("Failed to decode message");
        }
    }
}
