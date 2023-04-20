$folderPath = "assets/tests"
$exe = $args[0]
Get-ChildItem -Path $folderPath -Include Segmented_* -Recurse | Remove-Item
Remove-Item report.txt -ErrorAction SilentlyContinue
$files = Get-ChildItem -Path $folderPath -Filter *.obj

foreach ($file in $files) {
	echo serial $file.Name
    & $exe serial $file
	echo parallel $file.Name
    & $exe parallel $file
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