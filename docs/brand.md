# The mark

A colon and two lines of speech.

Every line of a scriba document opens with a name, then a colon, then what that
person said. The colon is the moment the words get attached to somebody, which is
the whole job of the tool and the one thing it does that a transcription app does
not. So the mark is that punctuation rather than a picture of sound.

`docs/img/mark.svg` is the drawing. `macapp/Tools/make-icon.swift` is the same
drawing in Core Graphics, which builds the application icon; the site header and
the favicon carry it inline. Change the SVG and change those two with it.

Two things were tried and rejected, and both are easy to fall back into.

A waveform. It is what every audio tool on earth uses, it says nothing about this
one, and it does not survive being sixteen pixels wide.

Three lines instead of two, with the dots spaced to match them. That is a
bulleted-list icon, which is a different piece of software. The dots sit a
colon's own distance apart, close, while the lines are spread. That difference is
what makes the mark read as punctuation.

## Colour

| | | |
|---|---|---|
| Ink | `#0E0D0C` | the plate the mark sits on, and the page |
| Amber | `#E0A458` | the colon, and one accent per screen |
| Paper | `#EDE8DE` | the words |

The greys carry a little red and yellow. Neutral greys with no chroma read as
dead next to them.

Amber is the accent and it is spent carefully: it marks the thing being said, or
the one control that matters on a screen. Used on everything it stops meaning
anything.

## The name

Lowercase, always: `scriba`, never `Scriba` in running text. The application
bundle is `Scriba.app` because macOS capitalises application names and arguing
with that only produces an odd-looking Dock.

In the wordmark the name carries the colon, `scriba:`, with the colon in amber.
It says what the product does without needing a tagline underneath it.
