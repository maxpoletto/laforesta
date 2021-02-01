Sub Refactor5()
'
' Refactor5 Macro
'

'
    Columns("A:B").EntireColumn.Select
    Selection.Insert Shift:=xlToRight, CopyOrigin:=xlFormatFromLeftOrAbove
    ActiveCell.Offset(1, 4).Range("A1").Select
    Selection.Copy
    ActiveCell.Offset(0, -4).Columns("A:A").Rows("1:150").Select
    ActiveSheet.Paste
    ActiveCell.Columns("A:A").EntireColumn.EntireColumn.AutoFit
    Application.CutCopyMode = False
    Cells.Find(What:="abete", After:=ActiveCell, LookIn:=xlValues, LookAt:= _
        xlPart, SearchOrder:=xlByRows, SearchDirection:=xlNext, MatchCase:=False) _
        .Activate
    ActiveCell.Offset(-1, -4).Range("A1").Select
    Selection.Copy
    ActiveCell.Offset(0, -3).Range("A1:A30").Select
    ActiveSheet.Paste
    ActiveCell.Offset(2, 0).Range("A1").Select
    Cells.Find(What:="abete", After:=ActiveCell, LookIn:=xlValues, LookAt:= _
        xlPart, SearchOrder:=xlByRows, SearchDirection:=xlNext, MatchCase:=False) _
        .Activate
    ActiveCell.Offset(-1, -4).Range("A1").Select
    Application.CutCopyMode = False
    Selection.Copy
    ActiveCell.Offset(0, -3).Range("A1:A20").Select
    ActiveSheet.Paste
    ActiveCell.Offset(2, 0).Range("A1").Select
    Cells.Find(What:="abete", After:=ActiveCell, LookIn:=xlValues, LookAt:= _
        xlPart, SearchOrder:=xlByRows, SearchDirection:=xlNext, MatchCase:=False) _
        .Activate
    ActiveCell.Offset(-1, -4).Range("A1").Select
    Application.CutCopyMode = False
    Selection.Copy
    ActiveCell.Offset(0, -3).Range("A1:A21").Select
    ActiveSheet.Paste
    ActiveCell.Offset(2, 0).Range("A1").Select
    Cells.Find(What:="abete", After:=ActiveCell, LookIn:=xlValues, LookAt:= _
        xlPart, SearchOrder:=xlByRows, SearchDirection:=xlNext, MatchCase:=False) _
        .Activate
    ActiveCell.Offset(-1, -4).Range("A1").Select
    Application.CutCopyMode = False
    Selection.Copy
    ActiveCell.Offset(0, -3).Range("A1:A23").Select
    ActiveSheet.Paste
    ActiveCell.Columns("A:A").EntireColumn.EntireColumn.AutoFit
    If ActiveSheet.Index = Worksheets.Count Then
    Worksheets(1).Select
    Else
    ActiveSheet.Next.Select
    End If
End Sub
