const fs = require('fs');
const args = process.argv.slice(2);
const os = require('os');
const cpus = os.cpus();
const memory = Math.floor(os.totalmem() / 1024 / 1024 / 1024 * 10) / 10;
let files = args;
files = files.map(f => ({name: f, content: fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : null}));

let file = files[0];
let modelFile = file.name.replace('.txt', '.obj');
let modelFileSize = fs.statSync(modelFile).size / 1024 / 1024;
let createdRawTime = fs.statSync(file.name).birthtime;
const createdTime = `${createdRawTime.getFullYear()}-${(createdRawTime.getMonth() + 1).toString().padStart(2, '0')}-${createdRawTime.getDate().toString().padStart(2, '0')} ${createdRawTime.getHours().toString().padStart(2, '0')}:${createdRawTime.getMinutes().toString().padStart(2, '0')}:${createdRawTime.getSeconds().toString().padStart(2, '0')}`;
let output = files[1];
let mode = file.name.match(/(?<=Segmented_)[a-zA-Z]+(?=_)/)[0];
let model = file.name.match(/(?<=Segmented_.+_).+(?=\.)/)[0];
let cpu = cpus[0].model.trim();
let threads = cpus.length;

let fileSplit = file.content.replace(/\r/g, '').split('\n');
let text = '';
for (let routineNumber = 0; routineNumber < fileSplit.length; routineNumber++) {
    let line = fileSplit[routineNumber];
    let lineSplit = line.split(':');
    if (lineSplit.length < 2) {
        continue;
    }
    let time = parseFloat(lineSplit.pop().trim());
    let routine = lineSplit.join().replace(/  - /gi, '').trim();
    text += [mode, model, modelFileSize, routine, routineNumber + 1, cpu, threads, memory, time, createdTime].map(e => typeof e == 'string' ? JSON.stringify(e) : e).join(',') + '\n';
}
if (output.content) {
    output.content += text;
} else {
    output.content = 'Mode,Model,ModelFileSize(MB),Routine,RoutineNumber,CPU,Threads,Memory(GB),Time,CreatedTime\n' + text;
}
fs.writeFileSync(output.name, output.content, 'utf8');