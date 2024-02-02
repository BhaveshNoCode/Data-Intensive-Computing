var express = require("express");
var app = express();
var bodyparser = require("body-parser");
var session = require("express-session");
var path = require("path");
var fs = require("fs");
var multer = require("multer")


let storage = multer.diskStorage({
    destination: './to_pred/',

    filename: function (req, file, cb) {
        cb(null, "excel.xlsx")
        // cb(null, file.originalname.replace(path.extname(file.originalname), "") + path.extname(file.originalname))
        // cb(null, file.originalname.replace('excel.xlsx'))
    }
})

let upload = multer({ storage: storage });


app.use(bodyparser.json());
app.use(bodyparser.urlencoded({ extended: false }));


app.use(session({
    secret: 'D1C_PR0J3CT',
    resave: false,
    saveUninitialized: true
}));



app.get("/", (req, res) => {
    console.log(__dirname)
    res.send("Server UP!");
});


app.get("/test", (req, res) => {
    const spawn = require("child_process").spawn;
    const py_process = spawn('python',["test.py", 54,1,3,42,6,1,9256,1999,0.812,1466,37,0.762,0.216,1,2,4,0]);
    py_process.stdout.on('data', (data) => {
        pred = data.toString();
        res.send("Prediction from Python File: "+pred);
    });
});

app.get("/test2", (req, res) => {
    console.log("/test2");
    const spawn = require("child_process").spawn;
    const py_process = spawn('python',["excel.py", './to_pred/excel.xlsx']);
    py_process.stdout.on('data', (data) => {
        pred = data.toString();
        arr_pred = JSON.parse(pred);
        arr_predStr = []
        arr_pred.forEach((curVal) => {
            if(curVal == 0){
                arr_predStr.push("Retention")
            } else if(curVal == 1){
                arr_predStr.push("Attrition")
            }
        });
        console.log("Sent!");
        // res.send("Prediction from Python File: "+arr_predStr);
        res.send(arr_predStr);
    });
});

app.post("/predict", (req,res) => {
    console.log("/predict")
    var form_data = req.body.info;
    const spawn = require("child_process").spawn;
    form_data.splice(0, 0, "test.py");
    console.log(form_data)
    // const py_process = spawn('python',["test.py", 54,1,3,42,6,1,9256,1999,0.812,1466,37,0.762,0.216,1,2,4,0]);
    const py_process = spawn('python',form_data);
    py_process.stdout.on('data', (data) => {
        pred = data.toString();
        pred = Number(pred)
        console.log(pred)
        predS = ''
        if(pred == 0){
            predS = 'Retention'
        } else if(pred == 1){
            predS = 'Attrition'
        }
        var to_send = {
            text: "Prediction from Machine Learning Model: "+predS
        }
        res.send(to_send);
    });
});

app.get("/pie", (req, res)=> {
    const img = "pie.jpg";
    const imgPath = path.join(__dirname, img);
    fs.exists(imgPath, (exists) => {
        if(exists) res.sendFile(imgPath);
        else res.send("Null");
    });
});

app.get("/bar", (req, res)=> {
    const img = "attrition_plots.png";
    const imgPath = path.join(__dirname, img);
    fs.exists(imgPath, (exists) => {
        if(exists) res.sendFile(imgPath);
        else res.send("Null");
    });
});


app.post("/upload", upload.single('file'), (req, res) => {
    console.log('File Uploaded Successfully! ', req.file.filename);
    res.status(200);
    res.send({info: 1});
});


// app.post("upload", (req, res, next) => {
//     var fstream;
//     console.log(req.busboy);
//     req.pipe(req.busboy);
//     req.busboy.on('file', function (fieldname, file, filename) {
//         console.log("Uploading: " + filename);
//         fstream = fs.createWriteStream(__dirname + '/' + filename);
//         file.pipe(fstream);
//         fstream.on('close', function () {    
//             console.log("Upload Finished of " + filename);              
//             res.redirect('back');
//         });
//     });
// });



app.use(express.static(path.join(__dirname, '/dicproject/dist/dicproject/browser')));

app.get("*", (req,res) => {
    res.sendFile(path.join(__dirname, '/dicproject/dist/dicproject/browser/index.html'));
});











const server = app.listen(process.env.PORT || 8123, process.env.IP, (req,res) => {
  console.log("Server Started!");
});