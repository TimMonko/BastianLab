// Put all files to be processed in correct folder OUTSIDE Google Drive
// Open Image using Bio-formats windowless importer
// Select PSD95 Image, if needed enhance contrast
// Image --> Adjust --> Brightness/Contrast (Control + Shift + C)
// Run this Macro
title = getTitle();
save_title = title + "_cropped.tif";
print(save_title);
close("\\Others"); //Closes all images except for the front image. 

run("Enhance Contrast", "saturated=0.35");
setTool("polygon");
waitForUser("draw polygon");

run("Add Selection...");
rename(save_title);

// Save via control+S  ... no need for changing title, folder will be in same location as original image