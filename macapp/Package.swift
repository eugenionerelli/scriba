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
        )
    ]
)
