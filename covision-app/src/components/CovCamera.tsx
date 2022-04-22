import { forwardRef } from "react";
import Webcam from "react-webcam";

const CovCamera: React.ForwardRefRenderFunction<Webcam> = (_, ref) => {
  return (
    <Webcam
      ref={ref}
      videoConstraints={{
        facingMode: "environment",
      }}
      width={"100%"}
      height={"100%"}
      style={{
        objectFit: "cover",
     }}
    />
  );
};

export default forwardRef(CovCamera);
