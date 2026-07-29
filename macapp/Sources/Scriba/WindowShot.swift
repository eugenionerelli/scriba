import AppKit

/// Saves a PNG of the app's own window, for documentation.
///
/// `screencapture` and every other screen grabber needs the screen recording
/// permission, which is a large thing to hand out for the sake of a README. An app
/// asking for an image of its own window needs nothing: it already owns the pixels.
/// Nothing outside this window can end up in the frame, which is also the point.
///
/// Off unless SCRIBA_SHOTS names a folder, so the menu item does not exist in a
/// normal build:
///
///     SCRIBA_SHOTS=~/dev/scriba/docs/img ./Scriba.app/Contents/MacOS/Scriba
///
/// then File > Save window as PNG, or ⌥⌘P.
enum WindowShot {

    static var isEnabled: Bool { directory != nil }

    static var directory: URL? {
        guard let raw = ProcessInfo.processInfo.environment["SCRIBA_SHOTS"],
              !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
    }

    @discardableResult
    static func save(named name: String? = nil) -> URL? {
        guard let dir = directory,
              let window = NSApp.keyWindow ?? NSApp.windows.first(where: \.isVisible)
        else { return nil }

        guard let data = png(of: window) else { return nil }

        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent((name ?? nextName()) + ".png")
        do {
            try data.write(to: url)
            NSSound(named: "Grab")?.play()
            return url
        } catch {
            NSSound.beep()
            return nil
        }
    }

    /// PNG of one window, from inside the process that owns it.
    ///
    /// Asking the window server for the window's own image is the only version of
    /// this that comes back complete. Drawing the view hierarchy by hand
    /// (`cacheDisplay`) skips whatever the compositor is responsible for, and on a
    /// NavigationSplitView that is the entire sidebar: it came out as a white
    /// rectangle with no rows in it.
    private static func png(of window: NSWindow) -> Data? {
        let id = CGWindowID(window.windowNumber)
        guard let image = CGWindowListCreateImage(
            .null, .optionIncludingWindow, id, [.boundsIgnoreFraming, .bestResolution])
        else { return nil }
        let rep = NSBitmapImageRep(cgImage: image)
        return rep.representation(using: .png, properties: [:])
    }

    /// shot-01, shot-02, and so on. Overwriting the previous one silently is how
    /// you end up with a README that quietly lost a picture.
    private static func nextName() -> String {
        guard let dir = directory else { return "shot" }
        let taken = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        for n in 1...99 {
            let candidate = String(format: "shot-%02d", n)
            if !taken.contains(candidate + ".png") { return candidate }
        }
        return "shot"
    }
}
