package com.byteai.local;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int BG = Color.rgb(10, 16, 29);
    private static final int PANEL = Color.rgb(18, 28, 46);
    private static final int BOT = Color.rgb(26, 39, 59);
    private static final int USER = Color.rgb(24, 91, 83);
    private static final int TEXT = Color.rgb(236, 243, 250);
    private static final int MUTED = Color.rgb(151, 168, 188);
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView status;
    private ByteAgent agent;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().getDecorView().setSystemUiVisibility(0);
        buildUi();
        addMessage(false, "BYTE v0.2 is loading its local model.\n\nThis app runs without an internet connection. Its tiny neural model is an experiment, not a ChatGPT-level assistant.");
        status.setText("LOADING LOCAL MODEL…");
        worker.execute(() -> {
            try {
                ByteEngine engine = new ByteEngine(getApplicationContext());
                ByteAgent ready = new ByteAgent(getApplicationContext(), engine);
                runOnUiThread(() -> {
                    agent = ready;
                    status.setText("LOCAL · OFFLINE · READY");
                    addMessage(false, "I'm ready. Type /help to see my tools. Try: calculate 18 * (4 + 2), remember my project is BYTE, or recall.");
                });
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    status.setText("MODEL LOAD FAILED");
                    addMessage(false, "I couldn't load the bundled model.\n\n" + ex.getMessage());
                });
            }
        });
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setPadding(dp(16), dp(12), dp(16), dp(12));

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView logo = new TextView(this);
        logo.setText("B");
        logo.setTextColor(BG);
        logo.setTextSize(22);
        logo.setTypeface(Typeface.DEFAULT_BOLD);
        logo.setGravity(Gravity.CENTER);
        GradientDrawable logoBg = new GradientDrawable(GradientDrawable.Orientation.TL_BR,
                new int[]{Color.rgb(112, 225, 193), Color.rgb(58, 167, 157)});
        logoBg.setCornerRadius(dp(13));
        logo.setBackground(logoBg);
        header.addView(logo, new LinearLayout.LayoutParams(dp(46), dp(46)));

        LinearLayout titleColumn = new LinearLayout(this);
        titleColumn.setOrientation(LinearLayout.VERTICAL);
        titleColumn.setPadding(dp(12), 0, 0, 0);
        TextView title = new TextView(this);
        title.setText("BYTE");
        title.setTextColor(TEXT);
        title.setTextSize(21);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        TextView subtitle = new TextView(this);
        subtitle.setText("YOUR LOCAL AI EXPERIMENT");
        subtitle.setTextColor(MUTED);
        subtitle.setTextSize(9);
        subtitle.setLetterSpacing(0.12f);
        titleColumn.addView(title);
        titleColumn.addView(subtitle);
        header.addView(titleColumn, new LinearLayout.LayoutParams(0, -2, 1));
        status = new TextView(this);
        status.setText("STARTING…");
        status.setTextColor(MUTED);
        status.setTextSize(9);
        status.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        header.addView(status);
        root.addView(header, new LinearLayout.LayoutParams(-1, dp(54)));

        ViewLine divider = new ViewLine(this);
        root.addView(divider, new LinearLayout.LayoutParams(-1, dp(1)));

        scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.setClipToPadding(false);
        scroll.setVerticalScrollBarEnabled(false);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(0, dp(14), 0, dp(14));
        scroll.addView(messages, new ScrollView.LayoutParams(-1, -2));
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));

        TextView toolsLabel = new TextView(this);
        toolsLabel.setText("LOCAL TOOLS");
        toolsLabel.setTextColor(MUTED);
        toolsLabel.setTextSize(9);
        toolsLabel.setLetterSpacing(0.14f);
        toolsLabel.setPadding(0, dp(4), 0, dp(6));
        root.addView(toolsLabel);

        LinearLayout quick = new LinearLayout(this);
        quick.setOrientation(LinearLayout.HORIZONTAL);
        quick.setGravity(Gravity.CENTER_VERTICAL);
        quick.addView(quickButton("HELP", "/help"), weightedParams());
        quick.addView(quickButton("STATUS", "status"), weightedParams());
        quick.addView(quickButton("MEMORY", "recall"), weightedParams());
        root.addView(quick, new LinearLayout.LayoutParams(-1, dp(38)));

        LinearLayout composer = new LinearLayout(this);
        composer.setOrientation(LinearLayout.HORIZONTAL);
        composer.setGravity(Gravity.CENTER_VERTICAL);
        composer.setPadding(dp(10), dp(5), dp(5), dp(5));
        GradientDrawable composerBg = rounded(PANEL, 18);
        composer.setBackground(composerBg);
        input = new EditText(this);
        input.setSingleLine(false);
        input.setMaxLines(4);
        input.setTextColor(TEXT);
        input.setHintTextColor(MUTED);
        input.setHint("Message BYTE…");
        input.setTextSize(15);
        input.setPadding(dp(8), dp(8), dp(8), dp(8));
        input.setBackgroundColor(Color.TRANSPARENT);
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        composer.addView(input, new LinearLayout.LayoutParams(0, -2, 1));
        send = new Button(this);
        send.setText("SEND");
        send.setTextColor(BG);
        send.setTextSize(11);
        send.setTypeface(Typeface.DEFAULT_BOLD);
        send.setAllCaps(true);
        send.setPadding(dp(12), 0, dp(12), 0);
        GradientDrawable sendBg = rounded(Color.rgb(112, 225, 193), 14);
        send.setBackground(sendBg);
        composer.addView(send, new LinearLayout.LayoutParams(-2, dp(44)));
        LinearLayout.LayoutParams composerParams = new LinearLayout.LayoutParams(-1, -2);
        composerParams.topMargin = dp(10);
        root.addView(composer, composerParams);

        send.setOnClickListener(v -> submit());
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) { submit(); return true; }
            return false;
        });
        setContentView(root);
    }

    private LinearLayout.LayoutParams weightedParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(34), 1);
        params.setMargins(0, 0, dp(6), 0);
        return params;
    }

    private Button quickButton(String label, String command) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(9);
        button.setTextColor(TEXT);
        button.setAllCaps(true);
        button.setPadding(dp(4), 0, dp(4), 0);
        button.setBackground(rounded(PANEL, 10));
        button.setOnClickListener(v -> {
            input.setText(command);
            submit();
        });
        return button;
    }

    private void submit() {
        String message = input.getText().toString().trim();
        if (message.isEmpty()) return;
        input.setText("");
        addMessage(true, message);
        if (agent == null) {
            addMessage(false, "The model is still loading. Try again in a moment.");
            return;
        }
        send.setEnabled(false);
        status.setText("THINKING LOCALLY…");
        worker.execute(() -> {
            String response;
            try { response = agent.respond(message); }
            catch (Throwable ex) { response = "BYTE agent error: " + ex.getMessage(); }
            final String finalResponse = response;
            runOnUiThread(() -> {
                addMessage(false, finalResponse);
                send.setEnabled(true);
                status.setText("LOCAL · OFFLINE · READY");
            });
        });
    }

    private void addMessage(boolean fromUser, String text) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(fromUser ? Gravity.RIGHT : Gravity.LEFT);
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextColor(TEXT);
        bubble.setTextSize(14);
        bubble.setLineSpacing(dp(2), 1.04f);
        bubble.setPadding(dp(13), dp(11), dp(13), dp(11));
        bubble.setMaxWidth(getResources().getDisplayMetrics().widthPixels * 4 / 5);
        bubble.setBackground(rounded(fromUser ? USER : BOT, 16));
        row.addView(bubble, new LinearLayout.LayoutParams(-2, -2));
        LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(-1, -2);
        rowParams.bottomMargin = dp(10);
        messages.addView(row, rowParams);
        scroll.post(() -> scroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    private GradientDrawable rounded(int color, int radiusDp) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color);
        shape.setCornerRadius(dp(radiusDp));
        return shape;
    }

    private int dp(int value) { return (int)(value * getResources().getDisplayMetrics().density + 0.5f); }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    private static final class ViewLine extends android.view.View {
        ViewLine(Activity activity) { super(activity); setBackgroundColor(Color.rgb(37, 51, 71)); }
    }
}
