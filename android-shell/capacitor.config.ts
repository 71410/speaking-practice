import type { CapacitorConfig } from "@capacitor/cli";

const STREAMLIT_URL =
  "https://speaking-practice-bhpcjdevjsrfpx99jckqpy.streamlit.app/";

const config: CapacitorConfig = {
  appId: "com.speakingpractice.trainer",
  appName: "IELTS Trainer",
  webDir: "www",
  server: {
    url: STREAMLIT_URL,
    cleartext: false
  },
  android: {
    allowMixedContent: false
  }
};

export default config;
