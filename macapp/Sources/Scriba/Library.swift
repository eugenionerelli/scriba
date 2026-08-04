import Foundation
import UniformTypeIdentifiers

/// Finding recordings, and tidying the list afterwards.
///
/// Two jobs that turn out to be the same job. Somebody with a folder of voice
/// memos does not want to drag them in one at a time, and somebody who has done
/// that a few times does not want a list of two hundred rows either. So: pull in
/// a whole tree at once, and be able to take things out of the list again.
enum Library {

    // MARK: - finding

    /// Every recording under a folder, in the order somebody would read them.
    ///
    /// Depth first and sorted, so a tree of interviews comes in grouped by folder
    /// rather than in whatever order the filesystem felt like. Packages are
    /// skipped: a .app or a Logic project is a folder full of audio that nobody
    /// means to transcribe.
    static func recordings(under root: URL, limit: Int = 2_000) -> [(url: URL, collection: String)] {
        let manager = FileManager.default
        var found: [(URL, String)] = []

        guard let walker = manager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isPackageKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        for case let url as URL in walker {
            if found.count >= limit { break }
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isPackageKey])
            if values?.isPackage == true {
                walker.skipDescendants()
                continue
            }
            guard values?.isRegularFile == true, isRecording(url) else { continue }
            found.append((url, collection(of: url, under: root)))
        }

        return found
            .sorted { ($0.1, $0.0.lastPathComponent) < ($1.1, $1.0.lastPathComponent) }
            .map { (url: $0.0, collection: $0.1) }
    }

    /// Whether this file is something the engine can read.
    ///
    /// By declared type rather than by extension: a voice memo exported as
    /// `.m4a`, a WhatsApp `.opus` and a screen recording `.mov` all pass, and a
    /// `.wav.txt` does not. Files with no type known to the system fall back to
    /// the extension list, because a file copied off a Windows machine sometimes
    /// arrives without one.
    static func isRecording(_ url: URL) -> Bool {
        if let type = try? url.resourceValues(forKeys: [.contentTypeKey]).contentType {
            if type.conforms(to: .audio) || type.conforms(to: .audiovisualContent) { return true }
            if type.conforms(to: .data) == false { return false }
        }
        return knownExtensions.contains(url.pathExtension.lowercased())
    }

    /// The same list the engine keeps, for the files macOS cannot type.
    static let knownExtensions: Set<String> = [
        "m4a", "mp3", "wav", "aiff", "aif", "aac", "flac", "ogg", "opus", "wma",
        "mp4", "mov", "m4v", "mkv", "avi", "webm",
    ]

    /// The folder a recording sat in, relative to the one it was added from.
    ///
    /// Files directly inside the chosen folder have no collection: they are
    /// already grouped by having been chosen. Deeper ones carry the path, so
    /// "Interviews/March" stays distinct from "Interviews/April".
    static func collection(of url: URL, under root: URL) -> String {
        let rootParts = root.standardizedFileURL.pathComponents
        let parts = url.standardizedFileURL.deletingLastPathComponent().pathComponents
        guard parts.count > rootParts.count,
              Array(parts.prefix(rootParts.count)) == rootParts else { return "" }
        return parts.dropFirst(rootParts.count).joined(separator: "/")
    }
}
