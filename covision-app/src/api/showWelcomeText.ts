const getCookie = (cname: string) => {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) === 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

const welcomeText =
  'Wilkommen bei CoVision. Richten Sie Ihre Kamera auf eine Covid Test Kasette um das Ergebnis zu erfahren oder auf den Barcode der Verpackung wenn Sie den Test noch nicht begonnen haben.';

const COOKIE_NAME = "welcomeTextShown";
const showWelcomeText = () => {
  if (getCookie(COOKIE_NAME) !== "1") {
    window.alert(welcomeText);
    document.cookie = COOKIE_NAME + "=1";
  }
}

export default showWelcomeText;