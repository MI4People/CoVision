import {
  IonButton,
  IonCard,
  IonCardContent,
  IonContent,
  IonPage,
  IonText,
  IonSelect,
  IonSelectOption,
} from '@ionic/react';
import { useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import CovCamera from '../components/CovCamera';
import { TestResult } from '../api/runClassifierAnalysis';
import showWelcomeText from '../api/showWelcomeText';
import usePipeline from '../api/usePipeline';
import { getInstruction } from '../data/instructions';
import { useHistory } from 'react-router';
import { useTranslation } from 'react-i18next';
import i18next from 'i18next';
import cookies from 'js-cookie';

showWelcomeText();

// const languages = [
//   {
//     code: 'de',
//     name: 'Deutsch',
//     country_code: 'de',
//   },
//   {
//     code: 'en',
//     name: 'English',
//     country_code: 'en',
//   },
// ];

const Home: React.FC = () => {
  // const currentLanguageCode = cookies.get('i18next') || 'en';
  // const currentLanguage = languages.find((l) => l.code === currentLanguageCode);
  const { t } = useTranslation();

  const webcamRef = useRef<Webcam>(null);
  const { result, detectionScore, area, barcodeResult } = usePipeline(webcamRef) ?? {};
  const history = useHistory();

  useEffect(() => {
    if (!barcodeResult) return;

    let index = getInstruction(barcodeResult.codeResult?.code);
    if (index !== -1) {
      history.push('/testInstruction/' + index);
    }

    // document.body.dir = currentLanguage.dir || 'ltr'
    // document.title = t('app_title')
  }, [barcodeResult, history, t]);

  return (
    <IonPage>
      <IonContent fullscreen>
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            top: 0,
            paddingTop: 'env(safe-area-inset-top)',
            left: 0,
            right: 0,
            display: 'flex',
            justifyContent: 'center',
            background: 'linear-gradient(0deg, rgba(24,24,24,0) 0%, rgba(24,24,24,1) 100%)',
          }}
        >
          <img aria-hidden="true" style={{ height: 80 }} src="/assets/logo.png" alt="CoVision" />
        </div>
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            display: 'flex',
            justifyContent: 'center',
            padding: '10px',
          }}
        >
          <IonButton style={{ width: '150px', 'font-size': '14px' }} href="/privacyPolicy">
            {t('privacypolicy')}
          </IonButton>
          <IonButton style={{ width: '150px', 'font-size': '14px' }} href="https://www.mi4people.org/imprint">
            {t('imprint')}
          </IonButton>
          <IonSelect
            onIonChange={(e) => i18next.changeLanguage(e.detail.value)}
            style={{ background: '#fff' }}
            placeholder="Select Language"
          >
            <IonSelectOption value="en">EN</IonSelectOption>
            <IonSelectOption value="de">DE</IonSelectOption>
          </IonSelect>
        </div>

        <div
          style={{
            position: 'absolute',
            bottom: 60,
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
                  {detectionScore !== -1
                    ? 'Test detected, result ' + TestResult[result] + '.               '
                    : 'Please scan a test'}
                  {result === TestResult.Positive && <h2 role="alert">Please call 116 117 to schedule a PCR test.</h2>}
                </h2>
                {false && ( // debug info
                  <h2>
                    {detectionScore !== -1 ? 1 : 0} tests detected (highest score: {detectionScore}), result:{' '}
                    {TestResult[result]}
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
      </IonContent>
    </IonPage>
  );
};

export default Home;
