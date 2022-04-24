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
  'Welcome to CoVision. Point your camera at a Covid test cassette to find out the result. Point your camera at the barcode of the packaging if you want to get a fully guided test instruction.';

const COOKIE_NAME = "welcomeTextShown";

const sleep = (ms: number) => new Promise(resolve => { setTimeout(resolve, ms) });
const showWelcomeText = async () => {
  await sleep(500);
  if (getCookie(COOKIE_NAME) !== "1") {
    window.alert(welcomeText);
    document.cookie = COOKIE_NAME + "=1";
  }
}

export default showWelcomeText;