//! Which graphics API a run uses.

/// `ORRIN_BACKEND` names it, else the windowed editor's saved setting. Metal is
/// macOS only and the default on Apple silicon; everywhere else Vulkan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Vulkan,
    Metal,
}

impl BackendChoice {
    pub const ENV: &str = "ORRIN_BACKEND";

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

    /// `named` is `ORRIN_BACKEND`, `saved` the user's setting (`None` is Auto)
    /// and `metal` whether this machine can draw through Metal, asked only of
    /// a choice that is not the variable's.
    ///
    /// The `Err` is the variable's alone, for the caller to report before it
    /// falls back to Vulkan. Otherwise the backend, and what to say about a
    /// fallback.
    pub fn resolve(
        named: Option<&str>,
        saved: Option<Self>,
        metal: impl FnOnce() -> Result<(), String>,
    ) -> Result<(Self, Option<String>), String> {
        if named.is_some_and(|name| !name.trim().is_empty()) {
            // Named, it is honoured as named or refused: the renderer fails
            // loudly on a Metal it cannot start, so a measurement is never
            // labelled with a backend it did not run on.
            return Self::parse(named).map(|choice| (choice, None));
        }
        // A default or a saved choice this machine cannot honour falls back
        // instead, and says so. A saved one that refused would leave editing
        // the file by hand as the only way back into the editor.
        let choice = saved.unwrap_or(Self::DEFAULT);
        if choice != Self::Metal {
            return Ok((choice, None));
        }
        Ok(match metal().err() {
            None => (Self::Metal, None),
            Some(reason) if saved.is_some() => (
                Self::Vulkan,
                Some(format!(
                    "the saved backend Metal cannot start ({reason}); using Vulkan"
                )),
            ),
            Some(reason) => (
                Self::Vulkan,
                Some(format!("Metal is unavailable ({reason}); using Vulkan")),
            ),
        })
    }

    /// The choice the environment makes, warning and using Vulkan when it
    /// cannot be honoured.
    pub fn from_env() -> Self {
        Self::from_env_and(None)
    }

    /// [`resolve`](Self::resolve) for this process, logging what it says.
    pub fn from_env_and(saved: Option<Self>) -> Self {
        let named = std::env::var(Self::ENV).ok();
        match Self::resolve(named.as_deref(), saved, metal_available) {
            Ok((choice, warning)) => {
                if let Some(warning) = warning {
                    tracing::warn!("{warning}");
                }
                choice
            }
            Err(reason) => {
                tracing::warn!("{reason}; using Vulkan");
                Self::Vulkan
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
        }
    }
}

#[cfg(target_os = "macos")]
fn metal_available() -> Result<(), String> {
    crate::gfx::metal::MtlDriver::new().map(drop)
}

#[cfg(not(target_os = "macos"))]
fn metal_available() -> Result<(), String> {
    Err("it needs macOS".to_owned())
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

    fn available() -> Result<(), String> {
        Ok(())
    }

    fn unavailable() -> Result<(), String> {
        Err("no Metal device".to_owned())
    }

    #[test]
    fn the_environment_beats_the_setting_which_beats_the_default() {
        use BackendChoice::{Metal, Vulkan};
        let chosen = |named, saved| BackendChoice::resolve(named, saved, available);
        assert_eq!(chosen(Some("vulkan"), Some(Metal)), Ok((Vulkan, None)));
        assert_eq!(chosen(None, Some(Vulkan)), Ok((Vulkan, None)));
        assert_eq!(
            chosen(Some("  "), Some(Vulkan)),
            Ok((Vulkan, None)),
            "blank"
        );
        assert_eq!(chosen(None, None), Ok((BackendChoice::DEFAULT, None)));
    }

    /// A saved choice is somebody's preference from another day, perhaps
    /// another machine: it must not be able to stop the editor opening, so it
    /// falls back as Auto does and says so by name.
    #[test]
    fn a_saved_backend_this_machine_cannot_honour_falls_back_and_is_named() {
        let (backend, warning) =
            BackendChoice::resolve(None, Some(BackendChoice::Metal), unavailable).unwrap();
        assert_eq!(backend, BackendChoice::Vulkan);
        let warning = warning.expect("the fallback is said");
        assert!(warning.contains("Metal"), "{warning}");
    }

    /// The same name in the environment is a measurement's label, so it is
    /// never substituted: off macOS it is an error, and on macOS it is honoured
    /// without asking, leaving the renderer to refuse it loudly.
    #[test]
    fn the_same_name_in_the_environment_is_not_substituted() {
        let named = BackendChoice::resolve(Some("metal"), None, || -> Result<(), String> {
            panic!("a named backend is not probed")
        });
        if cfg!(target_os = "macos") {
            assert_eq!(named, Ok((BackendChoice::Metal, None)));
        } else {
            assert!(named.is_err(), "{named:?}");
        }
        assert!(
            BackendChoice::resolve(Some("dx12"), Some(BackendChoice::Vulkan), available).is_err()
        );
    }

    #[test]
    fn auto_on_a_machine_without_metal_is_vulkan_with_a_word_about_it() {
        let (backend, warning) = BackendChoice::resolve(None, None, unavailable).unwrap();
        assert_eq!(backend, BackendChoice::Vulkan);
        assert_eq!(
            warning.is_some(),
            BackendChoice::DEFAULT == BackendChoice::Metal,
            "{warning:?}"
        );
    }
}
