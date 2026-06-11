package com.speakingpractice.trainer;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebResourceResponse;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private static final String APP_URL =
        "https://speaking-practice-bhpcjdevjsrfpx99jckqpy.streamlit.app/";
    private static final String APP_HOST =
        "speaking-practice-bhpcjdevjsrfpx99jckqpy.streamlit.app";
    private static final int AUDIO_PERMISSION_REQUEST = 1000;
    private static final int FILE_CHOOSER_REQUEST = 1001;
    private static final String WEBVIEW_POLYFILLS =
        "if(!Object.hasOwn){Object.hasOwn=function(o,p){return Object.prototype.hasOwnProperty.call(Object(o),p)}};"
            + "if(typeof AbortSignal!=='undefined'&&!AbortSignal.timeout){AbortSignal.timeout=function(ms){var c=new AbortController();setTimeout(function(){c.abort()},ms);return c.signal}};\n";
    private static final String STREAMLIT_CLOUD_HEIGHT_FIX =
        "(function(){"
            + "function applyAndroidHeightFix(){"
            + "document.documentElement.style.height='100%';"
            + "document.documentElement.style.minHeight='100%';"
            + "document.body.style.margin='0';"
            + "document.body.style.height='100%';"
            + "document.body.style.minHeight='100%';"
            + "document.body.style.overflow='auto';"
            + "var root=document.getElementById('root');"
            + "if(root){root.style.height='100%';root.style.minHeight='100%';}"
            + "var frame=document.querySelector('iframe[src*=\"~/+/\"]');"
            + "if(frame){"
            + "var el=frame;"
            + "while(el&&el!==document.body){"
            + "el.style.height='100vh';"
            + "el.style.minHeight='100vh';"
            + "el.style.width='100%';"
            + "if(el.tagName==='IFRAME'){el.style.border='0';el.style.display='block';}"
            + "el=el.parentElement;"
            + "}"
            + "}"
            + "window.dispatchEvent(new Event('resize'));"
            + "}"
            + "applyAndroidHeightFix();"
            + "setTimeout(applyAndroidHeightFix,250);"
            + "setTimeout(applyAndroidHeightFix,1000);"
            + "})();";

    private WebView webView;
    private ValueCallback<Uri[]> filePathCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestAudioPermissionIfNeeded();
        buildLayout();
        configureWebView();
        webView.loadUrl(APP_URL);
    }

    private void buildLayout() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);

        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setOrientation(LinearLayout.HORIZONTAL);
        toolbar.setPadding(18, 14, 18, 14);
        toolbar.setBackgroundColor(Color.rgb(247, 248, 245));

        TextView title = new TextView(this);
        title.setText("IELTS Trainer");
        title.setTextColor(Color.rgb(29, 38, 33));
        title.setTextSize(16);
        title.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(
            0,
            LinearLayout.LayoutParams.WRAP_CONTENT,
            1
        );
        toolbar.addView(title, titleParams);

        Button browserButton = new Button(this);
        browserButton.setText("浏览器打开");
        browserButton.setAllCaps(false);
        browserButton.setOnClickListener(view -> openInBrowser());
        toolbar.addView(browserButton);

        webView = new WebView(this);
        root.addView(
            toolbar,
            new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(56)
            )
        );
        root.addView(
            webView,
            new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1
            )
        );
        setContentView(root);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void configureWebView() {
        WebView.setWebContentsDebuggingEnabled(true);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setJavaScriptCanOpenWindowsAutomatically(true);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setSupportZoom(false);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
        webView.clearCache(true);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE);
        }

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
                injectPolyfills(view);
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                injectPostLoadFixes(view);
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                return false;
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(
                WebView view,
                WebResourceRequest request
            ) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    WebResourceResponse patched = patchJavaScriptResponse(request);
                    if (patched != null) {
                        return patched;
                    }
                }
                return super.shouldInterceptRequest(view, request);
            }

            @Override
            public void onReceivedError(
                WebView view,
                WebResourceRequest request,
                WebResourceError error
            ) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && request.isForMainFrame()) {
                    showErrorPage(error.getDescription().toString());
                }
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> request.grant(request.getResources()));
            }

            @Override
            public boolean onShowFileChooser(
                WebView webView,
                ValueCallback<Uri[]> filePathCallback,
                WebChromeClient.FileChooserParams fileChooserParams
            ) {
                if (MainActivity.this.filePathCallback != null) {
                    MainActivity.this.filePathCallback.onReceiveValue(null);
                }
                MainActivity.this.filePathCallback = filePathCallback;
                try {
                    startActivityForResult(
                        fileChooserParams.createIntent(),
                        FILE_CHOOSER_REQUEST
                    );
                    return true;
                } catch (Exception e) {
                    MainActivity.this.filePathCallback = null;
                    return false;
                }
            }
        });
    }

    private void injectPolyfills(WebView view) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            view.evaluateJavascript(WEBVIEW_POLYFILLS, null);
        }
    }

    private void injectPostLoadFixes(WebView view) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            view.evaluateJavascript(WEBVIEW_POLYFILLS + STREAMLIT_CLOUD_HEIGHT_FIX, null);
        }
    }

    private WebResourceResponse patchJavaScriptResponse(WebResourceRequest request) {
        try {
            Uri uri = request.getUrl();
            if (
                uri == null
                    || !APP_HOST.equals(uri.getHost())
                    || uri.getPath() == null
                    || !uri.getPath().endsWith(".js")
            ) {
                return null;
            }

            URL url = new URL(uri.toString());
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(15000);
            connection.setReadTimeout(30000);
            connection.setRequestProperty("Cache-Control", "no-cache");

            try (InputStream input = connection.getInputStream()) {
                byte[] original = readAllBytes(input);
                byte[] prefix = WEBVIEW_POLYFILLS.getBytes(StandardCharsets.UTF_8);
                ByteArrayOutputStream output = new ByteArrayOutputStream(
                    prefix.length + original.length
                );
                output.write(prefix);
                output.write(original);
                return new WebResourceResponse(
                    "application/javascript",
                    "UTF-8",
                    new ByteArrayInputStream(output.toByteArray())
                );
            } finally {
                connection.disconnect();
            }
        } catch (Exception e) {
            return null;
        }
    }

    private byte[] readAllBytes(InputStream input) throws java.io.IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = input.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private void showErrorPage(String reason) {
        String html = "<html><body style='font-family:sans-serif;padding:24px;'>"
            + "<h2>页面暂时没有加载成功</h2>"
            + "<p>请检查网络，或点击顶部“浏览器打开”。</p>"
            + "<p style='color:#666;font-size:13px;'>"
            + reason
            + "</p></body></html>";
        webView.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null);
    }

    private void requestAudioPermissionIfNeeded() {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                new String[] { Manifest.permission.RECORD_AUDIO },
                AUDIO_PERMISSION_REQUEST
            );
        }
    }

    private void openInBrowser() {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(APP_URL));
        startActivity(intent);
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != FILE_CHOOSER_REQUEST || filePathCallback == null) {
            return;
        }

        Uri[] results = null;
        if (resultCode == RESULT_OK && data != null) {
            Uri dataUri = data.getData();
            if (dataUri != null) {
                results = new Uri[] { dataUri };
            }
        }
        filePathCallback.onReceiveValue(results);
        filePathCallback = null;
    }
}
