import React, { useEffect, useRef } from 'react';
import Quagga from 'quagga';

type ScannerProps = {
  target: Element | null | undefined;
  onDetected: (data: {
    codeResult: {
      code: string;
    };
  }) => void;
};

const Scanner: React.FC<ScannerProps> = ({ target, onDetected }) => {
  const onDetectedRef = useRef<ScannerProps['onDetected']>(onDetected);
  onDetectedRef.current = onDetected;
  useEffect(() => {
    if (!target) return undefined;
    console.log('init');
    Quagga.init(
      {
        inputStream: {
          type: 'LiveStream',
          target,
          constraints: {
            width: 640,
            height: 320,
            facingMode: 'environment',
          },
          //   area: { // defines rectangle of the detection/localization area
          //     top: "10%",    // top offset
          //     right: "10%",  // right offset
          //     left: "10%",   // left offset
          //     bottom: "10%"  // bottom offset
          //   },
        },
        locator: {
          halfSample: true,
          patchSize: 'large', // x-small, small, medium, large, x-large
          debug: {
            showCanvas: true,
            showPatches: false,
            showFoundPatches: false,
            showSkeleton: false,
            showLabels: false,
            showPatchLabels: false,
            showRemainingPatchLabels: false,
            boxFromPatches: {
              showTransformed: true,
              showTransformedBox: true,
              showBB: true,
            },
          },
        },
        numOfWorkers: 4,
        decoder: {
          readers: ['code_128_reader', 'ean_reader'],
          debug: {
            drawBoundingBox: true,
            showFrequency: true,
            drawScanline: true,
            showPattern: true,
          },
        },
        locate: true,
      },
      function (err) {
        if (err) {
          return console.log(err);
        }
        Quagga.start();
      }
    );
    const handleDetected = (data) => {
      onDetectedRef.current(data);
    };
    Quagga.onDetected(handleDetected);
    return () => {
      Quagga.offDetected(handleDetected);
    };
  }, [target]);

  return <div id="interactive" className="viewport" />;
};

export default Scanner;
