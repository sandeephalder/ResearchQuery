/**
 * Clerk's middleware, so `auth()` and `<SignedIn>` work on every route.
 *
 * Nothing is protected here. The backend is what enforces access — every route
 * but `/health` sits behind `require_user`, verified against Clerk server-side
 * — and a UI that hides a page it cannot actually protect is theatre. What this
 * gives is the session in the browser, so the app can show a sign-in button
 * instead of a failed request.
 */

import { clerkMiddleware } from "@clerk/nextjs/server";

export default clerkMiddleware();

export const config = {
  matcher: [
    // Everything except Next's internals and static files.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
