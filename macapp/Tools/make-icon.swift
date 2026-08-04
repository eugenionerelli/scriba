// Draws the scriba mark at any size and writes the PNGs an .icns is packed from.
//
// The mark is also in docs/img/mark.svg, and this is the same drawing in Core
// Graphics rather than a second design: a colon and two lines of speech, which is
// what every line of a scriba document looks like. It is drawn rather than
// converted because the converter available here, qlmanage, is a thumbnailer: it
// rendered the artwork at its intrinsic size and cropped the result, so the icon
// came out with two of its four corners square.
//
//   swiftc -O -parse-as-library make-icon.swift -o make-icon
//   ./make-icon <output-directory>

import AppKit
import CoreGraphics
import Foundation

struct Palette {
    static let ink = CGColor(red: 0x0E / 255, green: 0x0D / 255, blue: 0x0C / 255, alpha: 1)
    static let amber = CGColor(red: 0xE0 / 255, green: 0xA4 / 255, blue: 0x58 / 255, alpha: 1)
    static let paper = CGColor(red: 0xED / 255, green: 0xE8 / 255, blue: 0xDE / 255, alpha: 1)
}

/// One tile of the icon. Coordinates are the mark's own 64 unit square, scaled up.
func draw(size: Int) -> CGImage? {
    let space = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                              bytesPerRow: 0, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }

    // macOS insets its icons inside the tile rather than filling it edge to edge.
    // Filling the tile is the single thing that makes an icon look like it was
    // made by somebody who has never shipped a Mac application.
    let inset = CGFloat(size) * 0.09
    let side = CGFloat(size) - inset * 2
    let unit = side / 64.0
    func x(_ v: CGFloat) -> CGFloat { inset + v * unit }
    // Core Graphics counts from the bottom, the drawing from the top.
    func y(_ v: CGFloat) -> CGFloat { inset + (64 - v) * unit }

    ctx.setFillColor(Palette.ink)
    let plate = CGPath(roundedRect: CGRect(x: inset, y: inset, width: side, height: side),
                       cornerWidth: 14 * unit, cornerHeight: 14 * unit, transform: nil)
    ctx.addPath(plate)
    ctx.fillPath()

    ctx.setFillColor(Palette.amber)
    for centre in [CGFloat(29), CGFloat(39)] {
        ctx.fillEllipse(in: CGRect(x: x(20 - 4), y: y(centre + 4),
                                   width: 8 * unit, height: 8 * unit))
    }

    ctx.setFillColor(Palette.paper)
    for (top, width) in [(CGFloat(25), CGFloat(20)), (CGFloat(37), CGFloat(13))] {
        let rect = CGRect(x: x(32), y: y(top + 6), width: width * unit, height: 6 * unit)
        ctx.addPath(CGPath(roundedRect: rect, cornerWidth: 3 * unit,
                           cornerHeight: 3 * unit, transform: nil))
        ctx.fillPath()
    }

    return ctx.makeImage()
}

func write(_ image: CGImage, to url: URL) throws {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "make-icon", code: 1)
    }
    try data.write(to: url)
}

@main
struct Main {
    static func main() {
        guard CommandLine.arguments.count > 1 else {
            print("usage: make-icon <output-directory>"); exit(1)
        }
        let out = URL(fileURLWithPath: CommandLine.arguments[1])
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        // The names macOS expects inside an .iconset.
        let tiles: [(String, Int)] = [
            ("icon_16x16", 16), ("icon_16x16@2x", 32),
            ("icon_32x32", 32), ("icon_32x32@2x", 64),
            ("icon_128x128", 128), ("icon_128x128@2x", 256),
            ("icon_256x256", 256), ("icon_256x256@2x", 512),
            ("icon_512x512", 512), ("icon_512x512@2x", 1024),
        ]
        for (name, size) in tiles {
            guard let image = draw(size: size) else { print("failed at \(size)"); exit(2) }
            do { try write(image, to: out.appendingPathComponent("\(name).png")) }
            catch { print("could not write \(name): \(error)"); exit(3) }
        }
        print("wrote \(tiles.count) tiles to \(out.path)")
    }
}
