// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Scriba",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "Scriba",
            path: "Sources/Scriba",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The recording side is testable without anybody in the room: the audio
        // conversion, the file the pipeline is handed afterwards, and the two
        // objects the interface reads while a session runs. Those are also where
        // the failures were. The microphone itself is exercised by live-check,
        // which needs an audio file and a speech model and so is a command rather
        // than a test.
        .testTarget(
            name: "ScribaTests",
            dependencies: ["Scriba"],
            path: "Tests/ScribaTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
