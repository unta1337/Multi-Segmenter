$folderPath = "assets/tests"
Get-ChildItem -Path $folderPath -Include Segmented_* -Recurse | Remove-Item
Remove-Item report.txt
$files = Get-ChildItem -Path $folderPath -Filter *.obj

foreach ($file in $files) {
	echo serial $file.Name
    cmake-build-release/Release/Multi-Segmenter.exe serial $file
	echo parallel $file.Name
    cmake-build-release/Release/Multi-Segmenter.exe parallel $file
}

$serials = Get-ChildItem -Path $folderPath -Filter *serial*.txt
$parallels = Get-ChildItem -Path $folderPath -Filter *parallel*.txt

for ($i=0; $i -lt $serials.Count; $i++) {
    $serial = $serials[$i]
    $parallel = $parallels[$i]
    echo $parallel.Name >> report.txt
    node utils/LogTool/index.js $serial $parallel >> report.txt
}