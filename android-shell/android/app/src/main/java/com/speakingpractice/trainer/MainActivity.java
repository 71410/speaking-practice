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
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.mozilla.geckoview.GeckoRuntime;
import org.mozilla.geckoview.GeckoSession;
import org.mozilla.geckoview.GeckoView;

public class MainActivity extends Activity {
    private static final String APP_URL =
        "https://speaking-practice-bhpcjdevjsrfpx99jckqpy.streamlit.app/";
    private static final int AUDIO_PERMISSION_REQUEST = 1000;

    private GeckoSession geckoSession;
    private GeckoView geckoView;
    private boolean canGoBack = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestAudioPermissionIfNeeded();
        buildLayout();
        startGecko();
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
        toolbar.addView(
            title,
            new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1
            )
        );

        Button browserButton = new Button(this);
        browserButton.setText("浏览器打开");
        browserButton.setAllCaps(false);
        browserButton.setOnClickListener(view -> openInBrowser());
        toolbar.addView(browserButton);

        geckoView = new GeckoView(this);
        root.addView(toolbar);
        root.addView(
            geckoView,
            new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1
            )
        );
        setContentView(root);
    }

    private void startGecko() {
        GeckoRuntime runtime = GeckoRuntime.create(this);
        geckoSession = new GeckoSession();
        geckoSession.setNavigationDelegate(new GeckoSession.NavigationDelegate() {
            @Override
            public void onCanGoBack(GeckoSession session, boolean canGoBack) {
                MainActivity.this.canGoBack = canGoBack;
            }
        });
        geckoSession.open(runtime);
        geckoView.setSession(geckoSession);
        geckoSession.loadUri(APP_URL);
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
        if (geckoSession != null && canGoBack) {
            geckoSession.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (geckoSession != null) {
            geckoSession.close();
        }
        super.onDestroy();
    }
}
