# Course Control Panel Changes

This document records custom modifications made to the `course_control_panel.xml` configuration during a troubleshooting session with the Paparazzi Center.

## Problem

When launching the joystick (PS4 gamepad) from Paparazzi Centre, an error was reported: `-d 0 is unknown` and the XML configuration could not be found. The same command worked from a terminal.

### Root causes identified

1. **Argument formatting**
   - The XML used `<arg flag="-d 0"/>` which produced a single argument `"-d 0"`. `input2ivy` expects two separate tokens: `-d` followed by the device index.

2. **Working directory mismatch**
   - Centre started the joystick program from the user's home directory. `input2ivy` always prepends `PAPARAZZI_HOME/conf/joystick/` to whatever filename is given. Thus relative paths became `.../conf/joystick/conf/joystick/ps4_gamepad.xml` and failed.

## Fixes applied

1. **XML argument correction**  
   In `course_control_panel.xml`, joystick program entries were changed to:
   ```xml
   <program name="Joystick" command="sw/ground_segment/joystick/input2ivy">
     <arg flag="-ac" constant="@AIRCRAFT"/>
     <arg flag="ps4_gamepad.xml"/>          <!-- base filename only -->
     <arg flag="-d" constant="0"/>          <!-- separate flag and value -->
   </program>
   ```
   This ensures proper tokenization by the argument parser.

2. **Centre launcher update**  
   `ProgramWidget.start_program()` in `sw/supervision/python/program_widget.py` was modified to set the process working directory to `PAPARAZZI_HOME`. This makes relative paths behave exactly as when the command is run from the repo root. Example change:
   ```python
   self.process.setWorkingDirectory(utils.PAPARAZZI_HOME)
   ```

3. **Lint cleanup**  
   Minor type-ignore comments and an import fix were added to `program_widget.py` so the file has no Pylance errors.

## Usage notes

- Always split multi-part flags (e.g. `-d 0`, `-i 1`) into separate `<arg flag="..." constant="..."/>` elements when editing control panels.
- If a custom joystick XML is used, just specify the filename (not a full path) and rely on `PAPARAZZI_HOME/conf/joystick` as the base location.
- The changes above only affect `course_control_panel.xml`; other control panel files remain unchanged.

## Result

After applying these adjustments, launching the session from Paparazzi Centre successfully starts `input2ivy`, detects the PS4 gamepad (`"Sony Computer Entertainment Wireless controller"` on device 0), and broadcasts correctly.

Keep this markdown file as a reminder for future joystick configuration work.
