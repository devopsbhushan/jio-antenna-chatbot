package com.jio.antennachatbot

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.KeyEvent
import android.webkit.*
import android.widget.ProgressBar
import android.widget.RelativeLayout
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar

    private val CHATBOT_URL =
        "https://chatbot-input-database.s3.ap-south-1.amazonaws.com/ui/index.html"

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Root layout
        val layout = RelativeLayout(this)
        layout.setBackgroundColor(Color.WHITE)

        // Progress bar
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
        @Suppress("DEPRECATION")
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
        }

        // Progress bar updates
        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                progressBar.progress = newProgress
                progressBar.visibility =
                    if (newProgress < 100) android.view.View.VISIBLE
                    else android.view.View.GONE
            }
        }

        // Stay in-app — don't open external browser
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
                if (request.isForMainFrame) {
                    view.loadData(
                        """
                        <html><body style="font-family:sans-serif;text-align:center;
                          padding:40px;color:#555;background:#fff">
                        <h2 style="color:#d50000">📡 Connection Error</h2>
                        <p>Unable to reach the chatbot.<br>
                        Please check your internet connection.</p>
                        <button onclick="location.reload()"
                          style="padding:12px 28px;background:#d50000;color:white;
                                 border:none;border-radius:24px;font-size:16px;
                                 margin-top:16px;cursor:pointer">
                          Retry
                        </button></body></html>
                        """.trimIndent(),
                        "text/html", "UTF-8"
                    )
                }
            }
        }

        layout.addView(webView)
        setContentView(layout)
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

    override fun onPause()  { super.onPause();  webView.onPause()  }
    override fun onResume() { super.onResume(); webView.onResume() }
}
