import AVFoundation
import Testing

@testable import Scriba

/// Tests for the recording side of the app.
///
/// What can be tested without a person in the room is everything except the
/// microphone itself: the audio conversion, the file the pipeline will later be
/// handed, and the two objects the interface reads while a recording runs.
/// Those are also where the failures were. Pushing a tap buffer into a converter
/// built for another format traps inside the audio layer rather than raising,
/// and an unclosed file has no length in its header, so every tool afterwards
/// reads a recording of unknown duration.
///
/// The microphone is exercised by `live-check`, which feeds a wav through the
/// same streaming path at wall-clock pace. That one needs an audio file and a
/// speech model, so it is a command rather than a test.
struct RecorderTests {

    /// The format the microphone hands over on this machine, or a common stand-in.
    static func tapFormat(rate: Double = 48_000, channels: AVAudioChannelCount = 1)
        -> AVAudioFormat {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate,
                      channels: channels, interleaved: false)!
    }

    static func fileFormat() -> AVAudioFormat {
        AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000,
                      channels: 1, interleaved: true)!
    }

    static func tone(_ format: AVAudioFormat, seconds: Double, amplitude: Float = 0.3)
        -> AVAudioPCMBuffer {
        let frames = AVAudioFrameCount(format.sampleRate * seconds)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<Int(format.channelCount) {
            for i in 0..<Int(frames) {
                let t = Double(i) / format.sampleRate
                buffer.floatChannelData![channel][i] = amplitude * Float(sin(2 * .pi * 220 * t))
            }
        }
        return buffer
    }

    @Test("a 44.1 kHz microphone is resampled to the file's 48 kHz")
    func resamples() throws {
        let from = Self.tapFormat(rate: 44_100)
        let to = Self.fileFormat()
        let converter = try #require(AVAudioConverter(from: from, to: to))
        let input = Self.tone(from, seconds: 1.0)

        let output = try #require(Recorder.convert(input, with: converter, to: to))
        #expect(output.format.sampleRate == 48_000)
        #expect(output.format.commonFormat == .pcmFormatInt16)
        // A second in, a second out, less the resampler's priming: the filter holds
        // back about 3000 frames on its first call and gives them back on the next
        // one. That is why the recorder keeps one converter for the whole session
        // rather than building one per buffer, and why the test below, which runs
        // twenty buffers through, comes out much closer to the ideal.
        let ideal = 48_000.0
        #expect(Double(output.frameLength) > ideal * 0.9)
        #expect(Double(output.frameLength) <= ideal)
    }

    @Test("a stereo microphone comes out mono, and not silent")
    func mixesDown() throws {
        let from = Self.tapFormat(rate: 48_000, channels: 2)
        let to = Self.fileFormat()
        let converter = try #require(AVAudioConverter(from: from, to: to))

        let output = try #require(Recorder.convert(Self.tone(from, seconds: 0.5),
                                                   with: converter, to: to))
        #expect(output.format.channelCount == 1)
        let samples = UnsafeBufferPointer(start: output.int16ChannelData![0],
                                          count: Int(output.frameLength))
        #expect(samples.contains { abs($0) > 1000 })
    }

    @Test("the same converter is reusable buffer after buffer")
    func reusable() throws {
        // A fresh converter per buffer resets the resampler's filter state and the
        // seam is audible, so the recorder keeps one for the session. This checks
        // that keeping one actually works rather than degrading after the first.
        let from = Self.tapFormat()
        let to = Self.fileFormat()
        let converter = try #require(AVAudioConverter(from: from, to: to))
        var total: AVAudioFrameCount = 0
        for _ in 0..<20 {
            let out = try #require(Recorder.convert(Self.tone(from, seconds: 0.1),
                                                    with: converter, to: to))
            total += out.frameLength
        }
        #expect(abs(Double(total) - 2.0 * 48_000) < 4_000)
    }

    @Test("a closed file reports the length it was written with")
    func fileHasDuration() throws {
        // The failure this stands for: quitting mid-recording without closing the
        // file leaves a WAV header that says nothing, and ffprobe answers N/A when
        // the pipeline asks how long the recording is.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("recording.wav")
        let format = Self.fileFormat()
        let from = Self.tapFormat()
        let converter = try #require(AVAudioConverter(from: from, to: format))

        var file: AVAudioFile? = try AVAudioFile(forWriting: url, settings: format.settings,
                                                 commonFormat: .pcmFormatInt16,
                                                 interleaved: true)
        for _ in 0..<10 {
            let out = try #require(Recorder.convert(Self.tone(from, seconds: 0.2),
                                                    with: converter, to: format))
            try file?.write(from: out)
        }
        file = nil          // this is what finalises the header

        let reread = try AVAudioFile(forReading: url)
        let seconds = Double(reread.length) / reread.fileFormat.sampleRate
        #expect(abs(seconds - 2.0) < 0.1)
        #expect(reread.fileFormat.channelCount == 1)
    }
}

@MainActor
struct LiveTextTests {

    @Test("a finished utterance joins the settled text and clears the guess")
    func commits() {
        let live = LiveText()
        live.hypothesise("buon")
        live.commit("Buongiorno.")
        #expect(live.settled == "Buongiorno.")
        #expect(live.guess.isEmpty)

        live.commit("Come stai?")
        #expect(live.settled == "Buongiorno. Come stai?")
    }

    @Test("empty results are ignored rather than adding blank space")
    func ignoresEmpty() {
        let live = LiveText()
        live.commit("   ")
        live.commit("\n")
        #expect(live.settled.isEmpty)
    }

    @Test("the hypothesis is published at most ten times a second")
    func coalesces() {
        // Not a nicety. A consumer that redrew on every result pushed the
        // displayed text from three seconds behind the speaker to seven, and the
        // gap grew for the length of the session.
        let live = LiveText()
        live.hypothesise("uno")
        #expect(live.guess == "uno")
        live.hypothesise("uno due")
        #expect(live.guess == "uno", "a second update inside the window is dropped")
    }

    @Test("reset clears everything, including a previous failure")
    func resets() {
        let live = LiveText()
        live.commit("qualcosa")
        live.note(unavailable: "no model")
        live.reset()
        #expect(live.settled.isEmpty)
        #expect(live.unavailable == nil)
    }
}

@MainActor
struct LiveMeterTests {

    @Test("the meter starts at zero and takes the first reading")
    func firstReading() {
        let meter = LiveMeter()
        #expect(meter.peak == 0)
        meter.update(peak: 0.4, seconds: 1.0)
        #expect(meter.peak == 0.4)
        #expect(meter.seconds == 1.0)
    }

    @Test("readings inside the same tenth of a second are dropped")
    func coalesces() {
        let meter = LiveMeter()
        meter.update(peak: 0.4, seconds: 1.0)
        meter.update(peak: 0.9, seconds: 1.05)
        #expect(meter.peak == 0.4)
    }
}

struct KeychainTests {

    @Test("what Swift writes is what the engine reads")
    func roundTrip() throws {
        // Against a throwaway service name: the real one holds the user's token
        // and a test has no business touching it. What this checks is that a
        // generic password written through the Security framework is the shape
        // `security find-generic-password -s <service> -w` returns, which is how
        // config.py reads it.
        let service = "scriba-tests-\(UUID().uuidString)"
        defer { Keychain.forget(service: service) }

        #expect(Keychain.hasToken(service: service) == false)
        #expect(Keychain.save("hf_not_a_real_token", service: service) == nil)
        #expect(Keychain.hasToken(service: service))

        let read = Process()
        read.executableURL = URL(fileURLWithPath: "/usr/bin/security")
        read.arguments = ["find-generic-password", "-s", service, "-w"]
        let pipe = Pipe()
        read.standardOutput = pipe
        try read.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        read.waitUntilExit()
        let value = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(value == "hf_not_a_real_token")
    }

    @Test("an empty token is refused rather than stored")
    func refusesEmpty() {
        let service = "scriba-tests-\(UUID().uuidString)"
        defer { Keychain.forget(service: service) }
        #expect(Keychain.save("   ", service: service) != nil)
        #expect(Keychain.hasToken(service: service) == false)
    }
}
