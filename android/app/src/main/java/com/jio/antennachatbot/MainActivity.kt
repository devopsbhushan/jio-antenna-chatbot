package com.jio.antennachatbot

import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.view.KeyEvent
import android.webkit.*
import android.widget.ProgressBar
import android.widget.RelativeLayout
import android.graphics.Color

class MainActivity : Activity() {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar

    // ── PUT YOUR LAMBDA / S3 URL HERE ──────────────────────────────────────
    private val CHATBOT_URL = "https://chatbot-input-database.s3.ap-south-1.amazonaws.com/ui/index.html"
    // ───────────────────────────────────────────────────────────────────────

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Root layout
        val layout = RelativeLayout(this)
        layout.setBackgroundColor(Color.WHITE)

        // Progress bar shown while page loads
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        progressBar.id = android.R.id.progress
        progressBar.max = 100
        val pbParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT, 8
        )
        progressBar.layoutParams = pbParams
        layout.addView(progressBar)

        // WebView
        webView = WebView(this)
        val wvParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT,
            RelativeLayout.LayoutParams.MATCH_PARENT
        )
        wvParams.addRule(RelativeLayout.BELOW, android.R.id.progress)
        webView.layoutParams = wvParams

        // Settings
        webView.settings.apply {
            javaScriptEnabled        = true
            domStorageEnabled        = true
            databaseEnabled          = true
            loadWithOverviewMode     = true
            useWideViewPort          = true
            builtInZoomControls      = false
            displayZoomControls      = false
            setSupportZoom(false)
            mixedContentMode         = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            cacheMode                = WebSettings.LOAD_DEFAULT
            userAgentString          = userAgentString + " JioAntennaChatbot/1.0"
        }

        // Show/hide progress bar on page load
        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                progressBar.progress = newProgress
                progressBar.visibility = if (newProgress < 100)
                    android.view.View.VISIBLE else android.view.View.GONE
            }
        }

        // Handle navigation within app (don't open external browser)
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView, request: WebResourceRequest
            ): Boolean {
                view.loadUrl(request.url.toString())
                return true
            }

            override fun onReceivedError(
                view: WebView, request: WebResourceRequest, error: WebResourceError
            ) {
                // Show friendly offline message
                view.loadData(
                    """
                    <html><body style="font-family:sans-serif;text-align:center;padding:40px;color:#555">
                    <h2>📡 Connection Error</h2>
                    <p>Unable to reach the chatbot server.<br>Please check your internet connection.</p>
                    <button onclick="location.reload()"
                      style="padding:12px 28px;background:#d50000;color:white;
                             border:none;border-radius:24px;font-size:16px;margin-top:16px">
                      Retry
                    </button>
                    </body></html>
                    """.trimIndent(),
                    "text/html", "UTF-8"
                )
            }
        }

        layout.addView(webView)
        setContentView(layout)

        // Load chatbot URL
        webView.loadUrl(CHATBOT_URL)
    }

    // Back button navigates WebView history
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack()) {
            webView.goBack()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    // Pause/resume WebView with activity lifecycle
    override fun onPause()  { super.onPause();  webView.onPause()  }
    override fun onResume() { super.onResume(); webView.onResume() }
}
