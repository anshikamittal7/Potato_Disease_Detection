import React from "react";
import { useDropzone } from "react-dropzone";

const HomeScreen = () => {
  const { acceptedFiles, getRootProps, getInputProps } = useDropzone();
  const files = acceptedFiles.map((file) => (
    <p key={file.path}>
      {file.path} - {file.size} bytes
    </p>
  ));

  return (
    <div className="home">
      <div className="backgroundimg"></div>
      <div className="dropbox">
        <div {...getRootProps({ className: "dropzone" })}>
          <input {...getInputProps()} />
          <button>Upload Leaf Image</button>
          <button>Upload Potato Image</button>
        </div>

        <aside>
          <p>{files}</p>
        </aside>
      </div>
    </div>
  );
};

export default HomeScreen;
