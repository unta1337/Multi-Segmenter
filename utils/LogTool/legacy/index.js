const fs = require('fs');
const args = process.argv.slice(2);
let files = args;
files = files.map(f => ({name: f, content: fs.readFileSync(f, 'utf8')}));

let serial = files[0];
let parallel = files[1];

let sb = '';
let serialSplit = serial.content.replace(/\r/g, '').split('\n');
let parallelSplit = parallel.content.replace(/\r/g, '').split('\n');
for(let i = 0; i < parallelSplit.length; i ++){
    let ps = parallelSplit[i];
    let pt = parseFloat(ps.split(':').pop().trim());
    let ss = serialSplit[i];
    let st = parseFloat(ss.split(':').pop().trim());
    if(!isNaN(pt)){
        sb += ps + ` (${Math.floor(st/pt * 10000)/10000}x)\n`;
    } else {
        sb += ps
    }
}
console.log(sb);