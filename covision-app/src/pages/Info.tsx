import { IonPage, IonText, IonContent } from '@ionic/react';
import { useTranslation } from 'react-i18next';

const Info: React.FC = () => {
  const { t } = useTranslation();
  return (
    <IonPage>
      <div
        style={{
          padding: '30px',
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        <IonContent fullscreen>
          <IonText>{t('infoText')}</IonText>
        </IonContent>
      </div>
    </IonPage>
  );
};

export default Info;
