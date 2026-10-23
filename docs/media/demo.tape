# CliCR terminal demo. Run inside docs/media/Dockerfile.vhs (see scripts/record_demo.ps1).
Output docs/media/demo.gif

Set Shell "bash"
Set Theme "Catppuccin Mocha"
Set Width 1200
Set Height 700
Set FontSize 18
Set Padding 20
Set TypingSpeed 45ms
# Local 7B answers can be slow; Wait fails after this long.
Set WaitTimeout 180s

Hide
Type "export CLICR_HOME=/tmp/clicr-home" Enter
Type "cd /clicr && clear" Enter
Show

Type "clicr index ." Sleep 500ms Enter
Wait+Screen /Indexed/
Sleep 2s

Hide
Type "clear" Enter
Show

Type 'clicr ask "How does clicr combine vector and keyword search?"' Sleep 500ms Enter
Wait+Screen /citations: \d+ valid/
Sleep 1s
Screenshot docs/media/ask.png
Sleep 4s

Hide
Type "clear" Enter
Show

Type 'clicr ask --explain "Which chunks does graph expansion add?"' Sleep 500ms Enter
Wait+Screen /citations: \d+ valid/
Sleep 1s
Screenshot docs/media/explain.png
Sleep 4s
