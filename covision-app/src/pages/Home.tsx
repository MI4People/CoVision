import { IonCard, IonCardContent, IonContent, IonPage, IonText } from '@ionic/react';
import { useRef } from 'react';
import Webcam from 'react-webcam';
import useYolov5Analysis, { Yolov5AnalysisResult } from '../api/useYolov5Analysis';
import CovCamera from '../components/CovCamera';

const getValidTestArea = (analysis: Yolov5AnalysisResult) => {
  if (!analysis.scores || !analysis.boxes) return null;
  for (let i = 0; i < 100; i += 1) {
    const hasTest = (analysis.scores[i] ?? -1) >= 0.6; // threshold
    if (hasTest) {
      const [topLeft, topRight, bottomLeft, bottomRight] = analysis.boxes.slice(i * 4, i * 4 + 4);
      return { score: analysis.scores[i], topLeft, topRight, bottomLeft, bottomRight };
    }
  }
};

const Home: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const analysis = useYolov5Analysis(webcamRef) ?? {};
  const test = getValidTestArea(analysis);
  console.log(test);

  return (
    <IonPage>
      <IonContent fullscreen>
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <IonCard>
            <IonCardContent>
              <IonText color="primary">
                <h1>{(test ? 1 : 0) + ' tests detected'}</h1>
              </IonText>
            </IonCardContent>
          </IonCard>
        </div>
        <CovCamera ref={webcamRef} />
      </IonContent>
    </IonPage>
  );
};

export default Home;
