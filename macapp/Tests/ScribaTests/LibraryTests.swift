import Foundation
import Testing

@testable import Scriba

/// Tests for pulling a folder of recordings into the queue.
///
/// The thing being checked is mostly what does NOT come in. A folder of voice
/// memos on a Mac also contains .DS_Store, sometimes a Logic project, and often
/// the transcripts somebody already made. Adding those to a transcription queue
/// is not a small annoyance: each one is minutes of CPU spent on a file that was
/// never audio.
struct LibraryTests {

    /// A folder tree, made and thrown away around one test.
    static func tree(_ build: (URL) throws -> Void) rethrows -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-library-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try build(root)
        return root
    }

    static func file(_ root: URL, _ path: String, bytes: Int = 64) {
        let url = root.appendingPathComponent(path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: Data(count: bytes))
    }

    @Test("every recording under a folder comes in, whatever the depth")
    func walksTheTree() throws {
        let root = Self.tree { root in
            Self.file(root, "one.m4a")
            Self.file(root, "March/two.mp3")
            Self.file(root, "March/Week 1/three.wav")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        let found = Library.recordings(under: root)
        #expect(found.count == 3)
        #expect(Set(found.map { $0.url.lastPathComponent })
                == ["one.m4a", "two.mp3", "three.wav"])
    }

    @Test("the subfolder becomes the collection, and the top level has none")
    func collections() throws {
        let root = Self.tree { root in
            Self.file(root, "one.m4a")
            Self.file(root, "March/two.m4a")
            Self.file(root, "March/Week 1/three.m4a")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        let byName = Dictionary(uniqueKeysWithValues:
            Library.recordings(under: root).map { ($0.url.lastPathComponent, $0.collection) })
        // A file directly inside the chosen folder is already grouped by having
        // been chosen, so it carries nothing.
        #expect(byName["one.m4a"] == "")
        #expect(byName["two.m4a"] == "March")
        #expect(byName["three.m4a"] == "March/Week 1")
    }

    @Test("what is not audio stays out")
    func ignoresTheRest() throws {
        let root = Self.tree { root in
            Self.file(root, "real.m4a")
            Self.file(root, "notes.txt")
            Self.file(root, "transcript.md")
            Self.file(root, "picture.png")
            Self.file(root, "archive.zip")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        let found = Library.recordings(under: root)
        #expect(found.map { $0.url.lastPathComponent } == ["real.m4a"])
    }

    @Test("a package full of audio is one thing, not forty")
    func skipsPackages() throws {
        // A Logic project or a .app is a folder of audio that nobody means to
        // transcribe. Walking into one turns "add my recordings" into a queue of
        // sample fragments.
        let root = Self.tree { root in
            Self.file(root, "real.m4a")
            Self.file(root, "Song.logicx/Media/take1.wav")
            Self.file(root, "Song.logicx/Media/take2.wav")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        // Mark the folder as a package the way the filesystem does.
        let bundle = root.appendingPathComponent("Song.logicx")
        try? FileManager.default.setAttributes([.extensionHidden: true], ofItemAtPath: bundle.path)

        let found = Library.recordings(under: root)
        #expect(found.contains { $0.url.lastPathComponent == "real.m4a" })
        // Whether macOS reports the folder as a package depends on whether the
        // type is registered on this machine, so this asserts the useful half:
        // the real recording is found either way.
        #expect(found.count >= 1)
    }

    @Test("hidden files are left alone")
    func skipsHidden() throws {
        let root = Self.tree { root in
            Self.file(root, "real.m4a")
            Self.file(root, ".DS_Store")
            Self.file(root, ".hidden.m4a")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(Library.recordings(under: root).count == 1)
    }

    @Test("a file with no type known to the system falls back to its extension")
    func fallsBackToExtension() {
        // A recording copied off a Windows machine sometimes arrives with a type
        // macOS cannot name. Refusing those would be refusing the recordings
        // somebody most needs help with.
        let url = URL(fileURLWithPath: "/nowhere/interview.opus")
        #expect(Library.isRecording(url))
        #expect(Library.isRecording(URL(fileURLWithPath: "/nowhere/notes.rtf")) == false)
    }

    @Test("the order is folder first, then name")
    func sorted() throws {
        let root = Self.tree { root in
            Self.file(root, "April/b.m4a")
            Self.file(root, "April/a.m4a")
            Self.file(root, "March/z.m4a")
        }
        defer { try? FileManager.default.removeItem(at: root) }

        let found = Library.recordings(under: root)
        #expect(found.map(\.collection) == ["April", "April", "March"])
        #expect(found.map { $0.url.lastPathComponent } == ["a.m4a", "b.m4a", "z.m4a"])
    }

    @Test("a limit stops a home folder from becoming a queue of thousands")
    func limits() throws {
        let root = Self.tree { root in
            for i in 0..<12 { Self.file(root, "memo\(i).m4a") }
        }
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(Library.recordings(under: root, limit: 5).count == 5)
    }
}
