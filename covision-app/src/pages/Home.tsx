import { IonCard, IonCardContent, IonContent, IonPage, IonText } from '@ionic/react';
import { useRef } from 'react';
import Webcam from 'react-webcam';
import CovCamera from '../components/CovCamera';
import { TestResult } from '../api/runClassifierAnalysis';
import showWelcomeText from '../api/showWelcomeText';
import usePipeline from '../api/usePipeline';
import Scanner from '../components/Scanner';
import { getInstruction } from '../data/instructions';
import { useHistory } from 'react-router';

showWelcomeText();

const Home: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const { result, classificationScore, detectionScore, area } = usePipeline(webcamRef) ?? {};
  const history = useHistory();

  return (
    <IonPage>
      <IonContent fullscreen>
        <div
          style={{
            position: 'absolute',
            top: 'env(safe-area-inset-top)',
            left: 0,
            right: 0,
            display: 'flex',
            justifyContent: 'center',
            background: 'linear-gradient(0deg, rgba(24,24,24,0) 0%, rgba(24,24,24,1) 100%);',
          }}
        >
          <img style={{ height: 80 }} src="/assets/logo.png" alt="CoVision" />
        </div>
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
              <IonText style={{ color: '#fff' }}>
                <h2 role="alert">
                  {detectionScore !== -1 ? 'Test detected, result ' + TestResult[result] + '. ' : 'Please scan a test'}
                </h2>
                {result === TestResult.Positive && <h2 role="alert">Please call 116 117 to schedule a PRC test.</h2>}
                {false && ( // debug info
                  <h2>
                    {detectionScore !== -1 ? 1 : 0} tests detected (highest score: {detectionScore}), result:{' '}
                    {TestResult[result]} (score: {classificationScore})
                  </h2>
                )}
              </IonText>
            </IonCardContent>
          </IonCard>
        </div>

        {area && (
          <div
            style={{
              position: 'absolute',
              borderWidth: 4,
              borderColor: 'red',
              borderStyle: 'solid',
              borderRadius: 8,
              zIndex: 10000,
              top: area.top * 100 + '%',
              bottom: (1 - area.bottom) * 100 + '%',
              left: area.left * 100 + '%',
              right: (1 - area.right) * 100 + '%',
            }}
          ></div>
        )}

        <CovCamera ref={webcamRef} />

        <Scanner
          target={webcamRef.current?.video}
          onDetected={(data) => {
            let index = getInstruction(data.codeResult.code);
            if (index !== -1) {
              history.push('/testInstruction/' + index);
            }
          }}
        />
      </IonContent>
    </IonPage>
  );
};

export default Home;
