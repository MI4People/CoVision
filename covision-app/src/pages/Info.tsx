import { IonPage, IonText } from '@ionic/react';
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
        <IonText>{t('welcome')}</IonText>
      </div>
    </IonPage>
  );
};

export default Info