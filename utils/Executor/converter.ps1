$folderPath = "assets/tests"
$exe = $args[0]
Get-ChildItem -Path $folderPath -Include Segmented_* -Recurse | Remove-Item
Remove-Item report.txt -ErrorAction SilentlyContinue
$files = Get-ChildItem -Path $folderPath -Filter *.obj

$tolerances = @(15, 30, 60, 90, 120, 150, 180)

foreach ($file in $files) {
    foreach ($tolerance in $tolerances) {
        echo serial $tolerance $file.Name
        & $exe serial $tolerance $file
        echo parallel $tolerance $file.Name
        & $exe parallel $tolerance $file
    }
}

$serials = Get-ChildItem -Path $folderPath -Filter *serial*.txt
$parallels = Get-ChildItem -Path $folderPath -Filter *parallel*.txt

for ($i=0; $i -lt $serials.Count; $i++) {
    $serial = $serials[$i]
    $parallel = $parallels[$i]
    echo $parallel.Name >> report.txt
    node utils/LogTool/converter.js $serial
    node utils/LogTool/index.js $parallel
}