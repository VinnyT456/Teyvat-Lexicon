use std::{thread, time::Duration};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, PhysicalPosition, Position, Size, WebviewUrl,
    WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

fn show_main_window(app: &AppHandle) {
    if let Some(quick) = app.get_webview_window("quick") {
        hide_quick_window(&quick);
    }
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn show_quick_window(app: &AppHandle, anchor: Option<(f64, f64)>) {
    if let Some(window) = app.get_webview_window("quick") {
        if window.is_visible().unwrap_or(false) {
            hide_quick_window(&window);
        } else {
            if let Some((center_x, below_y)) = anchor {
                position_quick_window(&window, center_x, below_y);
            }
            let _ = window.show();
            let _ = window.set_focus();
            let _ = window.emit("quick-window-focused", ());
        }
    }
}

fn position_quick_window(window: &tauri::WebviewWindow, center_x: f64, below_y: f64) {
    let monitor = window
        .monitor_from_point(center_x, below_y)
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten());
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let popup_width = 430.0 * scale;
    let popup_height = 720.0 * scale;
    let mut x = center_x - popup_width / 2.0;
    let mut y = below_y + 4.0 * scale;

    if let Some(monitor) = monitor {
        let area = monitor.work_area();
        let left = f64::from(area.position.x);
        let top = f64::from(area.position.y);
        let right = left + f64::from(area.size.width);
        let bottom = top + f64::from(area.size.height);
        x = x.clamp(left, (right - popup_width).max(left));
        y = y.clamp(top, (bottom - popup_height).max(top));
    }

    let _ = window.set_position(Position::Physical(PhysicalPosition::new(
        x.round() as i32,
        y.round() as i32,
    )));
}

fn hide_quick_window(window: &tauri::WebviewWindow) {
    let _ = window.emit("quick-window-hidden", ());
    let _ = window.hide();
}

#[tauri::command]
fn show_main_window_command(app: AppHandle) {
    show_main_window(&app);
}

#[tauri::command]
fn open_settings_command(app: AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn hide_settings_window_command(app: AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        let _ = window.hide();
    }
}

#[tauri::command]
fn quit_application_command(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
fn hide_quick_window_command(app: AppHandle) {
    if let Some(window) = app.get_webview_window("quick") {
        hide_quick_window(&window);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        show_quick_window(app, None);
                    }
                })
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            show_main_window_command,
            hide_quick_window_command,
            open_settings_command,
            hide_settings_window_command,
            quit_application_command
        ])
        .setup(|app| {
            let shortcut = "CommandOrControl+Shift+Space"
                .parse::<Shortcut>()
                .expect("valid global shortcut");
            if let Err(error) = app.global_shortcut().register(shortcut) {
                eprintln!("Could not register quick translate shortcut: {error}");
            }

            WebviewWindowBuilder::new(
                app,
                "quick",
                WebviewUrl::App("index.html?mode=quick".into()),
            )
            .title("Teyvat Lexicon · Quick Translate")
            .inner_size(430.0, 720.0)
            .min_inner_size(360.0, 560.0)
            .resizable(false)
            .decorations(false)
            .transparent(true)
            .shadow(true)
            .always_on_top(true)
            .skip_taskbar(true)
            .center()
            .visible(false)
            .build()?;

            WebviewWindowBuilder::new(
                app,
                "settings",
                WebviewUrl::App("index.html?mode=settings".into()),
            )
            .title("Teyvat Lexicon · Settings")
            .inner_size(500.0, 620.0)
            .min_inner_size(420.0, 520.0)
            .resizable(false)
            .center()
            .visible(false)
            .build()?;

            #[cfg(target_os = "macos")]
            {
                let quick_item =
                    MenuItem::with_id(app, "quick", "Quick Translate", true, None::<&str>)?;
                let desk_item =
                    MenuItem::with_id(app, "desk", "Open Translation Desk", true, None::<&str>)?;
                let separator = PredefinedMenuItem::separator(app)?;
                let quit_item =
                    MenuItem::with_id(app, "quit", "Quit Teyvat Lexicon", true, None::<&str>)?;
                let menu =
                    Menu::with_items(app, &[&quick_item, &desk_item, &separator, &quit_item])?;

                let tray_icon = app
                    .default_window_icon()
                    .expect("Tauri app icon is configured")
                    .clone();
                let _tray = TrayIconBuilder::new()
                    .icon(tray_icon)
                    .tooltip("Teyvat Lexicon · Quick Translate")
                    .menu(&menu)
                    .show_menu_on_left_click(false)
                    .on_menu_event(|app, event| match event.id.as_ref() {
                        "quick" => show_quick_window(app, None),
                        "desk" => show_main_window(app),
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click {
                            rect,
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } = event
                        {
                            if let (Position::Physical(position), Size::Physical(size)) =
                                (rect.position, rect.size)
                            {
                                let center_x = f64::from(position.x) + f64::from(size.width) / 2.0;
                                let below_y = f64::from(position.y) + f64::from(size.height);
                                show_quick_window(tray.app_handle(), Some((center_x, below_y)));
                            }
                        }
                    })
                    .build(app)?;
            }

            Ok(())
        })
        .on_window_event(|window, event| match event {
            WindowEvent::Focused(false) if window.label() == "quick" => {
                let window = window.clone();
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(120));
                    if window.is_visible().unwrap_or(false) && !window.is_focused().unwrap_or(false)
                    {
                        let _ = window.emit("quick-window-hidden", ());
                        let _ = window.hide();
                    }
                });
            }
            WindowEvent::CloseRequested { api, .. } => {
                api.prevent_close();
                if window.label() == "quick" {
                    let _ = window.emit("quick-window-hidden", ());
                }
                let _ = window.hide();
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
