use serde::{Serialize, Deserialize, Serializer, Deserializer};
use std::time::{SystemTime, Duration, UNIX_EPOCH};
use num_complex::Complex64;

/// Serializes Duration as seconds
pub fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    duration.as_secs_f64().serialize(serializer)
}

/// Deserializes Duration from seconds
pub fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = f64::deserialize(deserializer)?;
    Ok(Duration::from_secs_f64(secs))
}

/// Serializes SystemTime as duration since UNIX_EPOCH
pub fn serialize_time<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let duration = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    duration.as_secs_f64().serialize(serializer)
}

/// Deserializes SystemTime from duration since UNIX_EPOCH
pub fn deserialize_time<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = f64::deserialize(deserializer)?;
    let duration = Duration::from_secs_f64(secs);
    Ok(UNIX_EPOCH + duration)
}

/// Serializes Complex64 as [real, imag] array
pub fn serialize_complex<S>(complex: &Complex64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    [complex.re, complex.im].serialize(serializer)
}

/// Deserializes Complex64 from [real, imag] array
pub fn deserialize_complex<'de, D>(deserializer: D) -> Result<Complex64, D::Error>
where
    D: Deserializer<'de>,
{
    let [re, im]: [f64; 2] = Deserialize::deserialize(deserializer)?;
    Ok(Complex64::new(re, im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_time_serialization() {
        #[derive(Serialize, Deserialize)]
        struct Test {
            #[serde(serialize_with = "serialize_time")]
            #[serde(deserialize_with = "deserialize_time")]
            time: SystemTime,
        }

        let original = Test {
            time: SystemTime::now(),
        };

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: Test = serde_json::from_str(&serialized).unwrap();

        let diff = deserialized.time
            .duration_since(original.time)
            .unwrap_or_else(|e| e.duration());
        
        assert!(diff < Duration::from_millis(1));
    }

    #[test]
    fn test_complex_serialization() {
        #[derive(Serialize, Deserialize)]
        struct Test {
            #[serde(serialize_with = "serialize_complex")]
            #[serde(deserialize_with = "deserialize_complex")]
            complex: Complex64,
        }

        let original = Test {
            complex: Complex64::new(1.0, 2.0),
        };

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: Test = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original.complex, deserialized.complex);
    }
}
