import type { Metadata } from "next";
import { ClerkProvider, SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";

import { HealthBadge } from "@/components/HealthBadge";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchQuery",
  description: "Ask a corpus of arXiv papers, and get an answer that cites its passages.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>
          <header className="bar">
            <h1>ResearchQuery</h1>
            {/* Open route, so it reports before sign-in — which is what makes it
                useful: "is the server up" is the first question when a call
                fails, and it should not itself need a token. */}
            <HealthBadge />
            <span className="spacer" />
            <SignedOut>
              <SignInButton mode="modal">
                <button>Sign in</button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </header>
          <div className="shell">{children}</div>
        </body>
      </html>
    </ClerkProvider>
  );
}
