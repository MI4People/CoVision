import { IonPage, IonText, IonContent } from '@ionic/react';

const PrivacyPolicy: React.FC = () => {
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
          <IonText>
            <b>Datenschutzerklärung für die Nutzung des CoVision-Systems</b>
            <br />
            <br />
            <b>Stand: 01.09.2022</b>
            <br />
            <br />
            Personenbezogene Daten werden bei der Nutzung des CoVision-System erhoben, um eine funktionsfähige und
            insbesondere barrierefreie Nutzung von CoVision zu gewährleisten. CoVision dient der Klassifikation von
            CoVid-19-Schnelltests für sehbehinderte und blinde Menschen.
            <br />
            <br />
            Mit dieser Datenschutzerklärung informieren wir Sie über Art, Umfang, Zweck, Dauer und Rechtsgrundlage der
            Verarbeitung personenbezogener Daten.
            <br />
            <br />
            Verantwortliche Stelle für die Datenverarbeitung bei der Nutzung des CoVision-Systems im
            datenschutzrechtlichen Sinne ist die MI4People gemeinnützige GmbH, Winklsaß 49, 84088 Neufahrn in
            Niederbayern. Erreichbar telefonisch unter +49 176 24973790 oder per Mail unter{' '}
            <a href="mailto:info@mi4people.org">info@mi4people.org</a>
            <br />
            <br />
            Datenschutzbeauftragter ist Dr. Paul Springer. Er ist unter den genannten Kontaktangaben zu erreichen.
            <br />
            <br />
            Gegenüber der verantwortlichen Stelle können Nutzer <b>folgende Rechte geltend</b> machen:
            <br />
            - Das Recht auf Auskunft über die verarbeiteten Daten nach Art. 15 DSGVO;
            <br />
            - Das Recht auf Berichtigung oder Vervollständigung unrichtiger bzw. unvollständiger Daten nach Art. 16
            DSGVO;
            <br />
            - Das Recht auf unverzügliche Löschung der betreffenden Daten nach Art. 17 DSGVO sowie auf Einschränkung der
            Verarbeitung nach Maßgabe von Art. 18 DSGVO;
            <br />
            - Das Recht auf Erhalt und Übermittlung gespeicherter Daten nach Art. 20 DSGVO;
            <br />
            - Das Recht auf Beschwerde gegenüber der Aufsichtsbehörde nach Art. 77 DSGVO;
            <br />
            <br />
            Die Verantwortliche Stelle ist verpflichtet über die Erfüllung der genannten Rechte zu informieren. <br />
            <br />
            Bei der Nutzung von CoVision werden <b>folgende Daten</b> erhoben:
            <br />
            - IP-Adresse
            <br />
            - Daten der Endgeräte
            <br />
            - Ergebnis des Covid-19-Tests (positiv, negativ, ungültig). Hierbei handelt es sich um ein Gesundheitsdatum
            nach Art. 9 Abs. 1 DSGVO.
            <br />
            <br />
            Diese Daten werden nur temporär für die Dauer der Nutzung auf Ihrem Endgerät{' '}
            <b>gespeichert. Sie werden gelöscht</b>, sobald die Anwendung geschlossen und die Nutzung beendet wird.{' '}
            <br />
            <br />
            Die Verarbeitung erfolgt auf der <b>Rechtsgrundlage</b> von Art. 9 Abs. 1 Buchstaben a) und h) DSGVO. Eine
            Einwilligung kann jederzeit widerrufen werden, mit der Beendigung der Nutzung wird zugleich auch die
            Datenverarbeitung beendet.
            <br />
            <br />
            Die Richtigkeit und Zuverlässigkeit des Ergebnisses eines Covid-19-Tests oder dessen Anzeige kann nicht
            gewährleistet werden.
            <br />
            <br />
          </IonText>
        </IonContent>
      </div>
    </IonPage>
  );
};

export default PrivacyPolicy;
