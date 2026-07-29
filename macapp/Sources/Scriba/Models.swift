import Foundation

/// A conversation turn already attributed to a speaker.
struct Turn: Codable, Identifiable, Hashable {
    var id: String { "\(start)-\(speaker ?? "?")" }
    let speaker: String?
    let start: Double
    let end: Double
    let text: String
}

/// The outcome of matching a voice print against the registry.
struct VoiceMatch: Codable, Hashable {
    let name: String?        // certain: applied on its own
    let candidate: String?   // borderline: proposed, waiting for confirmation
    let score: Double
    let reason: String
}

/// The state of a job, exactly as `scriba info` writes it.
struct JobInfo: Codable {
    let jobDir: String
    let source: String
    let turns: [Turn]
    let outputs: [String]
    let dossier: String
    let audio: String
    let state: JobState

    enum CodingKeys: String, CodingKey {
        case jobDir = "job_dir", source, turns, outputs, dossier, audio, state
    }
}

struct JobState: Codable {
    var duration: Double?
    var language: String?
    var languageNote: String?
    var languageConfidence: Double?
    var names: [String: String]?
    var matches: [String: VoiceMatch]?
    var wordLevel: Bool?

    enum CodingKeys: String, CodingKey {
        case duration, language, names, matches
        case languageNote = "language_note"
        case languageConfidence = "language_confidence"
        case wordLevel = "word_level"
    }
}

/// A speaker found in the recording, the way the app shows it.
struct Speaker: Identifiable, Hashable {
    let id: String            // SPEAKER_00
    var name: String          // whatever the user types
    var suggested: String?    // proposed by the voice registry
    var score: Double
    var reason: String
    var speechSeconds: Double
    var turnCount: Int
    var previewStart: Double
    var previewEnd: Double
    var longestQuote: String

    var displayName: String {
        name.isEmpty ? Speaker.label(for: id) : name
    }

    /// SPEAKER_00 -> "Voice 1". pyannote labels start at zero and carry a leading
    /// zero; nobody reading the transcript needs to know that.
    static func label(for id: String) -> String {
        guard id.hasPrefix("SPEAKER_"),
              let n = Int(id.dropFirst("SPEAKER_".count)) else { return id }
        return "Voice \(n + 1)"
    }

    /// The registry picked out the voice and the name is already applied.
    var isConfirmedMatch: Bool {
        !(suggested ?? "").isEmpty && name == suggested
    }

    /// The registry proposes a name that nobody has confirmed yet.
    var pendingSuggestion: String? {
        guard let s = suggested, !s.isEmpty, name.isEmpty else { return nil }
        return s
    }
}

enum Phase: String {
    case idle          = "Waiting"
    case preparing     = "Preparing the audio"
    case detecting     = "Detecting the language"
    case transcribing  = "Transcribing"
    case aligning      = "Aligning the words"
    case diarizing     = "Separating the voices"
    case identifying   = "Matching against the voice registry"
    case exporting     = "Writing the files"
    case done          = "Done"
    case failed        = "Error"

    /// Phases are read off the lines the CLI prints. Those lines come out on stdout,
    /// not on stderr. Engine reads both streams for that reason: watching stderr
    /// alone is exactly how this stopped working the first time.
    static func from(logLine line: String) -> Phase? {
        let l = line.lowercased()
        if l.contains("audio:") { return .preparing }
        if l.contains("language:") { return .detecting }
        if l.contains("transcription:") { return .transcribing }
        // "alignment:" with the colon, not bare "alignment": the failure message
        // reads "alignment failed (…)", and matching that would light this phase up
        // precisely when alignment did not happen.
        if l.contains("alignment:") { return .aligning }
        if l.contains("diarization:") { return .diarizing }
        // "registry:" is printed on every run, matched or not. The arrow lines
        // are printed only when somebody was recognised, so on a first run with
        // an empty registry this stage used to be skipped over on screen.
        if l.contains("registry:") { return .identifying }
        if l.contains("voice ") && l.contains("→") { return .identifying }
        if l.contains("wrote") { return .exporting }
        return nil
    }
}

func timecode(_ seconds: Double) -> String {
    let s = max(0, Int(seconds.rounded()))
    return String(format: "%02d:%02d", s / 60, s % 60)
}

func humanDuration(_ seconds: Double) -> String {
    let s = max(0, Int(seconds.rounded()))
    if s < 60 { return "\(s) s" }
    return "\(s / 60) min \(s % 60) s"
}
