use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq)]
pub enum SpectrumError {
    EmptyFrequencyGrid,
    LengthMismatch {
        frequencies: usize,
        values: usize,
    },
    NonFiniteFrequency {
        index: usize,
    },
    FrequenciesNotStrictlyIncreasing {
        lower_index: usize,
    },
    NegativePsdFrequency {
        index: usize,
    },
    NonFiniteTransferValue {
        index: usize,
    },
    InvalidPsdValue {
        index: usize,
    },
    NonFiniteQuery,
    FrequencyOutsideRange {
        frequency: f64,
        minimum: f64,
        maximum: f64,
    },
}

impl fmt::Display for SpectrumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrequencyGrid => write!(formatter, "frequency grid must not be empty"),
            Self::LengthMismatch {
                frequencies,
                values,
            } => write!(
                formatter,
                "frequency grid has {frequencies} points but spectrum has {values} values"
            ),
            Self::NonFiniteFrequency { index } => {
                write!(formatter, "frequency at index {index} is not finite")
            }
            Self::FrequenciesNotStrictlyIncreasing { lower_index } => write!(
                formatter,
                "frequencies at indices {lower_index} and {} are not strictly increasing",
                lower_index + 1
            ),
            Self::NegativePsdFrequency { index } => {
                write!(formatter, "PSD frequency at index {index} is negative")
            }
            Self::NonFiniteTransferValue { index } => {
                write!(
                    formatter,
                    "transfer response at index {index} is not finite"
                )
            }
            Self::InvalidPsdValue { index } => write!(
                formatter,
                "PSD value at index {index} must be finite and nonnegative"
            ),
            Self::NonFiniteQuery => write!(formatter, "query frequency must be finite"),
            Self::FrequencyOutsideRange {
                frequency,
                minimum,
                maximum,
            } => write!(
                formatter,
                "frequency {frequency} is outside the sampled range [{minimum}, {maximum}]"
            ),
        }
    }
}

impl Error for SpectrumError {}
