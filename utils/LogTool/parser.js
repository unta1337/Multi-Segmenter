const fs = require('fs');
const os = require('os');
const {execSync} = require('child_process');
const path = require('path');

function getGPUs() {
    const output = execSync('nvidia-smi -L', {encoding: 'utf-8'});
    const gpuList = output.trim().split('\n');
    const gpuNames = gpuList.map((gpu) => {
        const regex = /GPU \d+: (.+) \(/;
        const match = gpu.match(regex);
        if (match && match.length > 1) {
            return match[1];
        }
        return null;
    }).filter((name) => name !== null);
    return gpuNames;
}

const {ArgumentParser} = require('argparse');
const {GoogleSpreadsheet} = require('google-spreadsheet');

const parser = new ArgumentParser({
    description: 'Multi-Segmenter Result Parser'
});

parser.add_argument('-f', '--file', {type: 'str', help: 'Result Text File'});
parser.add_argument('-e', '--gsemail', {type: 'str', help: 'Google Spreadsheet email'});
parser.add_argument('-k', '--gskey', {type: 'str', help: 'Google Spreadsheet key'});
parser.add_argument('-d', '--gsdoc', {type: 'str', help: 'Google Spreadsheet doc'});
parser.add_argument('-w', '--watch', {type: 'str', help: 'Watch mode path'});

const args = parser.parse_args();
const gpus = getGPUs();

async function getData(filePath) {
    const version = execSync('git rev-parse HEAD', {encoding: 'utf-8'}).trim();
    const commit = execSync('git log --pretty=\'format:%Creset%s\' --no-merges -1', {encoding: 'utf-8'}).trim();
    const branch = execSync('git rev-parse --abbrev-ref HEAD', {encoding: 'utf-8'}).trim();
    const extension = path.extname(filePath);
    const fileName = path.basename(filePath, extension);
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const fileContentSplit = fileContent.split('\n');
    const routines = fileContentSplit.map(line => {
        const lineSplit = line.split(':');
        let routine = lineSplit.slice(0, lineSplit.length - 1).join(':').trim();
        routine = routine.replace(/^- /, '');
        const time = parseFloat(lineSplit[lineSplit.length - 1]);
        if (!Number.isNaN(time)) {
            return {routine, time};
        } else {
            return null;
        }
    }).filter(r => r);

    const mode = fileName.match(/(?<=Segmented_)[a-zA-Z]+(?=_)/)[0];
    const model = fileName.match(/(?<=Segmented_.+_\d{1,4}.\d_).+(?=$)/)[0];
    let modelFilePath = filePath.split(/[\\/]/g).slice(0, -1).join('/') + '/' + model + '.obj';
    let modelFileSize = fs.statSync(modelFilePath).size;
    const tolerance = parseFloat(fileName.match(/(?<=Segmented_.+_).+(?=_)/)[0]);

    const cpus = os.cpus();
    const cpu = cpus[0].model.trim();
    const threads = cpus.length;
    const memory = os.totalmem();
    const createdAtRaw = fs.statSync(filePath).birthtime;
    const createdAt = `${createdAtRaw.getFullYear()}-${(createdAtRaw.getMonth() + 1).toString().padStart(2, '0')}-${createdAtRaw.getDate().toString().padStart(2, '0')} ${createdAtRaw.getHours().toString().padStart(2, '0')}:${createdAtRaw.getMinutes().toString().padStart(2, '0')}:${createdAtRaw.getSeconds().toString().padStart(2, '0')}`;

    return {
        mode,
        model,
        tolerance,
        routines,
        modelFileSize,
        cpu,
        threads,
        memory,
        gpus,
        createdAt,
        version,
        commit,
        branch
    };
}


!async function () {
    if (args.watch) {
        const lock = new Map();
        fs.watch(args.watch, async function (event, fileName) {
            if (fileName.endsWith('.txt')) {
                if (!lock.has(fileName)) {
                    lock.set(fileName, true);
                    setTimeout(async _ => {
                        const fullPath = `${path.resolve(args.watch)}/${fileName}`;
                        const serialFile = fullPath.replace(/Segmented_(cuda|parallel)/, 'Segmented_serial');
                        if (fs.existsSync(serialFile) && fullPath !== serialFile) {
                            const target = (await getData(fullPath)).routines;
                            const serial = (await getData(serialFile)).routines;
                            const maxLength = target.map(r => r.routine.length).reduce((a, c) => Math.max(a, c));
                            console.log(`[${fileName}] ${new Date()}`);
                            console.log(target.map(r => `\t${r.routine.padEnd(maxLength)}: ${r.time} MS (${Math.floor(serial.find(s => s.routine === r.routine).time / r.time * 10000) / 10000}x)`).join('\n'));
                        }
                        lock.delete(fileName);
                    }, 100);
                }
            }
        });
        return;
    }
    const data = await getData(args.file);
    console.log(JSON.stringify(data, null, 4));
    if (args.gsemail && args.gskey && args.gsdoc) {
        const doc = new GoogleSpreadsheet(args.gsdoc);
        await doc.useServiceAccountAuth({
            client_email: args.gsemail,
            private_key: Buffer.from(args.gskey, 'base64').toString('utf8'),
        });
        await doc.loadInfo();
        const sheet = doc.sheetsByIndex[0];
        await sheet.loadHeaderRow();
        const headers = sheet.headerValues;
        const list = data.routines.map((r, i) => ({
            ...data, ...r,
            routineNumber: i + 1,
            routines: undefined,
            gpus: gpus.join(', ')
        })).map(r => headers.map(k => r[k]));
        await sheet.addRows(list);
        console.log(`End upload: ${list.length} Rows`);
    }
}();
