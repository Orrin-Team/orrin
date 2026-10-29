//! Which graphics API a run uses.

/// `ORRIN_BACKEND` names it; Vulkan is the default and Metal is macOS only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Vulkan,
    Metal,
}

impl BackendChoice {
    /// What `ORRIN_BACKEND` asks for. A name that cannot be honoured is an error
    /// for the caller to report before it falls back to Vulkan, rather than a
    /// silent substitution a frame-time comparison would then be mislabelled by.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(Self::Vulkan);
        };
        match value.to_ascii_lowercase().as_str() {
            "vulkan" => Ok(Self::Vulkan),
            "metal" if cfg!(target_os = "macos") => Ok(Self::Metal),
            "metal" => Err("ORRIN_BACKEND=metal needs macOS".to_string()),
            other => Err(format!(
                "ORRIN_BACKEND={other} is not a backend (vulkan, metal)"
            )),
        }
    }

    /// The choice the environment makes, warning and using Vulkan when it
    /// cannot be honoured.
    pub fn from_env() -> Self {
        Self::parse(std::env::var("ORRIN_BACKEND").ok().as_deref()).unwrap_or_else(|reason| {
            tracing::warn!("{reason}; using Vulkan");
            Self::Vulkan
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_is_the_default() {
        assert_eq!(BackendChoice::parse(None), Ok(BackendChoice::Vulkan));
        assert_eq!(BackendChoice::parse(Some("  ")), Ok(BackendChoice::Vulkan));
    }

    #[test]
    fn names_are_case_and_space_insensitive() {
        assert_eq!(
            BackendChoice::parse(Some(" Vulkan ")),
            Ok(BackendChoice::Vulkan)
        );
    }

    #[test]
    fn an_unknown_name_is_reported_not_guessed() {
        let error = BackendChoice::parse(Some("dx12")).unwrap_err();
        assert!(error.contains("dx12"), "{error}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_is_chosen_by_name_on_macos() {
        assert_eq!(
            BackendChoice::parse(Some("metal")),
            Ok(BackendChoice::Metal)
        );
        assert_eq!(
            BackendChoice::parse(Some("METAL")),
            Ok(BackendChoice::Metal)
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_is_refused_off_macos() {
        assert!(BackendChoice::parse(Some("metal")).is_err());
    }
}
