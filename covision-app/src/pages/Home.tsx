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
import { useHistory } from 'react-router';
import { useTranslation } from 'react-i18next';
import i18next from 'i18next';

showWelcomeText();

const Home: React.FC = () => {
  const { t } = useTranslation();

  const webcamRef = useRef<Webcam>(null);
  const { result, detectionScore, area } = usePipeline(webcamRef) ?? {};
  const history = useHistory();

  useEffect(() => {}, [history, t]);

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
          aria-hidden="true"
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            display: 'flex',
            float: 'right',
            background: 'linear-gradient(0deg, rgba(24,24,24,0) 0%, rgba(24,24,24,1) 100%)',
          }}
        >
          <IonSelect
            onIonChange={(e) => i18next.changeLanguage(e.detail.value)}
            style={{ background: '#fff' }}
            placeholder={t('language')}
          >
            <IonSelectOption value="en">EN</IonSelectOption>
            <IonSelectOption value="de">DE</IonSelectOption>
          </IonSelect>
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
          <IonButton style={{ width: '150px', 'font-size': '14px' }} href={t('imprintLink')}>
            {t('imprint')}
          </IonButton>
          <a href="/info">
            <button style={{ width: '150px', height: '35px', background: 'blue' }}>{t('info')}</button>
          </a>
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
                    ? t('testDetected') + TestResult[result] + '.               '
                    : t('pleaseScan')}
                  {result === TestResult.Positive && <h2 role="alert">{t('pleaseCall')}</h2>}
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
