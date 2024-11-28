document
  .getElementById("extract-form")
  .addEventListener("submit", function (e) {
    e.preventDefault();
    const url = document.getElementById("url-input").value;
    const resBox = document.getElementById("responseData");
    const loader = document.getElementById("loader");

    // Show the loader while fetching data
    loader.style.display = "block";

    fetch(`http://localhost:3000/?url=${encodeURIComponent(url)}`)
      .then((response) => response.json())
      .then((data) => {
        // Assuming data.pythonOutput contains the string with dark pattern occurrences
        const darkPatternData = data.pythonOutput;
        resBox.innerText = darkPatternData;

        loader.style.display = "none";
      })
      .catch((error) => {
        console.error("Error:", error);
        // Hide the loader in case of an error
        loader.style.display = "none";
      });
  });

document
  .getElementById("extract-review-form")
  .addEventListener("submit",function(e){
    e.preventDefault()
    const review=document.getElementById("review-input").value
    const responsebox=document.getElementById("reviewresponse")
    const loader = document.getElementById("loader");
    loader.style.display = "block";
      fetch(`http://localhost:3000/review`,{
        method:"POST",
        headers: {
          "Content-Type": "application/json",
        },
        body:JSON.stringify({review:review})
      })
      .then((res)=>res.json())
      .then((data)=>{
        const output=data
        responsebox.innerText=output;
        loader.style.display = "none";
      })
      .catch((error) => {
        console.error("Error:", error);
        // Hide the loader in case of an error
        loader.style.display = "none";
      });
    

  })