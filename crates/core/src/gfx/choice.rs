//! Which graphics API a run uses.

/// `ORRIN_BACKEND` names it. Metal is macOS only and the default on Apple
/// silicon; everywhere else Vulkan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Vulkan,
    Metal,
}

impl BackendChoice {
    /// What an unset `ORRIN_BACKEND` means.
    pub const DEFAULT: Self = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Self::Metal
    } else {
        Self::Vulkan
    };

    /// What `ORRIN_BACKEND` asks for. A name that cannot be honoured is an error
    /// for the caller to report before it falls back to Vulkan, rather than a
    /// silent substitution a frame-time comparison would then be mislabelled by.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(Self::DEFAULT);
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
        let named = std::env::var("ORRIN_BACKEND").ok();
        let choice = Self::parse(named.as_deref()).unwrap_or_else(|reason| {
            tracing::warn!("{reason}; using Vulkan");
            Self::Vulkan
        });
        // A default that this machine cannot honour falls back; a named one
        // fails loudly at startup instead, so a comparison is never mislabelled.
        #[cfg(target_os = "macos")]
        if choice == Self::Metal
            && named.as_deref().is_none_or(|name| name.trim().is_empty())
            && let Err(reason) = crate::gfx::metal::MtlDriver::new()
        {
            tracing::warn!("Metal is unavailable ({reason}); using Vulkan");
            return Self::Vulkan;
        }
        choice
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
    fn an_unset_backend_is_the_platform_default() {
        assert_eq!(BackendChoice::parse(None), Ok(BackendChoice::DEFAULT));
        assert_eq!(BackendChoice::parse(Some("  ")), Ok(BackendChoice::DEFAULT));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_defaults_to_metal() {
        assert_eq!(BackendChoice::DEFAULT, BackendChoice::Metal);
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
