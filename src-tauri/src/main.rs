// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
  net::TcpStream,
  path::{Path, PathBuf},
  process::{Child, Command, Stdio},
  sync::{Arc, Mutex},
  thread,
  time::Duration,
};

use tauri::{Manager, WindowEvent};

/// Configuration
const STREAMLIT_PORT: u16 = 8590;
const STREAMLIT_URL: &str = "http://127.0.0.1:8590";

struct ServerState {
  child: Arc<Mutex<Option<Child>>>,
}

impl ServerState {
  fn new() -> Self {
    Self {
      child: Arc::new(Mutex::new(None)),
    }
  }
}

fn main() {
  tauri::Builder::default()
    // Updater plugin (enabled in tauri.conf.json)
    .plugin(tauri_plugin_updater::Builder::new().build())
    // Spawn Streamlit (if needed) before showing the window
    .setup(|app| {
      // If something already listens on the port (dev mode), don't spawn again.
      if !port_is_open(STREAMLIT_PORT) {
        match spawn_streamlit_sidecar(app) {
          Ok(child) => {
            // Store the handle for later cleanup
            let state = ServerState::new();
            {
              let mut guard = state.child.lock().unwrap();
              *guard = Some(child);
            }
            app.manage(state);

            // Wait until the server is reachable (up to ~60s)
            wait_for_port(STREAMLIT_PORT, 120, Duration::from_millis(500));
          }
          Err(e) => {
            // If we failed to start Streamlit, show a best-effort error window.
            let _ = tauri::api::dialog::message(
              Some(app.get_window("main").unwrap_or_else(|| app.create_window("err", tauri::WindowUrl::default(), Default::default()).unwrap())),
              "Startup error",
              format!("Failed to start the embedded Streamlit server:\n\n{e}\n\nMake sure Python & dependencies are bundled and the start script exists."),
            );
          }
        }
      }
      // Optionally, navigate (in case the configured URL wasn't loaded yet)
      if let Some(win) = app.get_window("main") {
        let _ = win.eval(&format!("window.location.replace('{STREAMLIT_URL}');"));
      }
      Ok(())
    })
    // Ensure we kill the child process when the window is closed
    .on_window_event(|window, event| {
      if let WindowEvent::CloseRequested { .. } = event {
        let app = window.app_handle();
        if let Some(state) = app.try_state::<ServerState>() {
          if let Ok(mut guard) = state.child.lock() {
            if let Some(mut child) = guard.take() {
              #[cfg(target_os = "windows")]
              {
                // Best-effort: kill the process; batch will terminate its children in most cases.
                let _ = child.kill();
              }
              #[cfg(not(target_os = "windows"))]
              {
                let _ = child.kill();
              }
            }
          }
        }
        // Give the OS a moment to reap the process
        thread::sleep(Duration::from_millis(150));
      }
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

/// Try to connect to 127.0.0.1:port. If connection works, the port is open.
fn port_is_open(port: u16) -> bool {
  TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// Wait until the port is open or timeout (attempts * delay each).
fn wait_for_port(port: u16, attempts: usize, delay: Duration) {
  for _ in 0..attempts {
    if port_is_open(port) {
      return;
    }
    thread::sleep(delay);
  }
}

/// Spawns the start script with flags to pin the Streamlit port.
/// Looks for `start_windows.bat` on Windows and `start.sh` on Unix.
/// Tries multiple locations to account for bundling differences.
fn spawn_streamlit_sidecar(app: &tauri::AppHandle) -> std::io::Result<Child> {
  let script = if cfg!(target_os = "windows") {
    "start_windows.bat"
  } else {
    "start.sh"
  };

  let script_path = find_script_path(app, script).ok_or_else(|| {
    std::io::Error::new(
      std::io::ErrorKind::NotFound,
      format!("Could not locate start script: {script}"),
    )
  })?;

  // Build command
  #[cfg(target_os = "windows")]
  let mut cmd = {
    // Pass args to the batch file; the script should forward arguments after `--` to Streamlit.
    let mut c = Command::new("cmd");
    c.arg("/C")
      .arg(&script_path)
      .arg("--")
      .arg("--server.port")
      .arg(STREAMLIT_PORT.to_string())
      .arg("--server.headless")
      .arg("true")
      .arg("--browser.gatherUsageStats")
      .arg("false");
    c
  };

  #[cfg(not(target_os = "windows"))]
  let mut cmd = {
    // Use bash -lc to tolerate spaces in paths and executable bits.
    let quoted = format!(
      "\"{}\" -- --server.port {} --server.headless true --browser.gatherUsageStats false",
      script_path.display(),
      STREAMLIT_PORT
    );
    let mut c = Command::new("bash");
    c.arg("-lc").arg(quoted);
    c
  };

  if let Some(dir) = script_path.parent() {
    cmd.current_dir(dir);
  }

  // Quiet stdout/stderr (you can flip to inherit() while debugging)
  cmd.stdout(Stdio::null()).stderr(Stdio::null());

  let child = cmd.spawn()?;
  Ok(child)
}

/// Try common locations for the script:
/// - app resources (tauri resources)
/// - next to the executable (externalBin)
/// - current working dir (dev fallback)
#[allow(clippy::needless_pass_by_value)]
fn find_script_path(app: &tauri::AppHandle, filename: &str) -> Option<PathBuf> {
  // 1) Resources (works when bundled as a resource)
  if let Some(p) = app.path_resolver().resolve_resource(filename) {
    if p.exists() {
      return Some(p);
    }
  }

  // 2) Next to the executable (works for externalBin)
  if let Ok(exe) = std::env::current_exe() {
    if let Some(dir) = exe.parent() {
      let candidate = dir.join(filename);
      if candidate.exists() {
        return Some(candidate);
      }

      // macOS app bundle: .../MacOS/<binary> ; resources often live one dir up in ../Resources
      #[cfg(target_os = "macos")]
      {
        let candidate2 = dir.join("../Resources").join(filename);
        if candidate2.exists() {
          return Some(candidate2);
        }
      }
    }
  }

  // 3) CWD (dev convenience)
  let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
  let candidate = cwd.join(filename);
  if candidate.exists() {
    return Some(candidate);
  }

  None
}
