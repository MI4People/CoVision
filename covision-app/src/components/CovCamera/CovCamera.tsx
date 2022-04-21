import {  useCallback, useEffect, useRef } from "react";
import Webcam from "react-webcam";

const useEverySecond = (callback: () => void) => {
  useEffect(() => {
    const handle = setInterval(callback, 1000);
    return () => (clearInterval(handle));
  }, [callback]);
};

const CovCamera: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);

  useEverySecond(useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    console.log(imageSrc);
  }, []))

  return (
    <Webcam
      ref={webcamRef}
      videoConstraints={{
        facingMode: "environment",
      }}
      width={400}
      height={400}
    />
  );
};

export default CovCamera;
