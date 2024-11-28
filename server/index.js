import puppeteer from "puppeteer";
import express from "express";
import cors from "cors";
import fs from "fs";
import fileReader from "./file_reader.js";
import pythonFile from "./pythonCall.js";
import { spawn } from "child_process";

const app = express();

app.use(cors());
app.use(express.json());

const processUrl = async (url, res) => {
  try {
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto(url);

    const special_characters =
      /[\[\],"\{\}\(\)\<\>\/\?;\:|\`~\!\@\#\$\%\^\&\*\_\-\+\=\.\n\t ]/g;

    const texts = await page.$$eval(
      "p, h1, h2, h3, h4, h5, h6, li, a",
      (elements, special_characters) =>
        elements
          .map((element) => element.textContent.replace(special_characters, ""))
          .filter(
            (text) =>
              text.trim() !== "" &&
              text !== '""' &&
              text !== "\n\t\t\t\t\\n\t\t\t\t"
          ), // Remove empty strings and specific strings
      special_characters
    );
    

    await browser.close();

    // Run file clean process
    await fileCleanProcess(texts);


    // Call the Python process
    const pythonOutput = await pythonFile.runPythonProcess();

    res.json({
      texts,
      pythonOutput,
      message: `Output returned from python file: ${pythonOutput}`,
    });
  } catch (error) {
    console.error("Error:", error);
    // Ensure no further code is executed after sending a response
    return res.status(500).send("Internal Server Error");
  }
};

const fileCleanProcess = (texts) => {
  try {
    const cleanedTexts = texts.join("\n");
    const filePath = "./output.txt";
    fs.writeFileSync(filePath, cleanedTexts, "utf8");
    console.log("File clean process completed");
  } catch (error) {
    console.error(error);
    throw new Error("File clean process failed");
  }
};

app.get("/", async (req, res) => {
  const url = req.query.url || 'https://leetcode.com/';
  if (!url) {
    return res.status(400).send("Bad Request: URL parameter is missing");
  }

  // Process the URL and run the model
  await processUrl(url, res);
});
app.post("/review",async(req,res)=>{
  const rev=req.body.review
  const process = spawn(
    "python",
    ["./reviewmodel.py",rev]
  );
  const message="";
  process.stdout.on('data',function(data){
    console.log(data.toString())
    res.json(data.toString())
  })
  process.stderr.on("data", (data) => {
    console.error(`Python Error: ${data}`);
    // console.log("Im here on python error");
  });
})

app.listen(3000, () => {
  console.log("Server running on port http://localhost:3000");
});
