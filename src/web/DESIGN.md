---
name: Vietnamese Credit Intelligence System
colors:
  surface: '#fcf8fa'
  surface-dim: '#dcd9db'
  surface-bright: '#fcf8fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f5'
  surface-container: '#f0edef'
  surface-container-high: '#eae7e9'
  surface-container-highest: '#e4e2e4'
  on-surface: '#1b1b1d'
  on-surface-variant: '#45464d'
  inverse-surface: '#303032'
  inverse-on-surface: '#f3f0f2'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e74'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#131b2e'
  on-primary-container: '#7c839b'
  inverse-primary: '#bec6e0'
  secondary: '#4b41e1'
  on-secondary: '#ffffff'
  secondary-container: '#645efb'
  on-secondary-container: '#fffbff'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#002113'
  on-tertiary-container: '#009668'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fd'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3f465c'
  secondary-fixed: '#e2dfff'
  secondary-fixed-dim: '#c3c0ff'
  on-secondary-fixed: '#0f0069'
  on-secondary-fixed-variant: '#3323cc'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#fcf8fa'
  on-background: '#1b1b1d'
  surface-variant: '#e4e2e4'
typography:
  display-lg:
    fontFamily: Be Vietnam Pro
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 60px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Be Vietnam Pro
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Be Vietnam Pro
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Be Vietnam Pro
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  data-tabular:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  container-max: 1280px
  gutter: 24px
---

## Brand & Style
The design system is engineered for a high-trust, high-precision FinTech environment. It targets financial analysts and consumers within the Vietnamese credit ecosystem, requiring a balance of authoritative professionalism and modern clarity.

The aesthetic follows **Modern Corporate Minimalism** with a focus on data density and legibility. It utilizes a layered approach where information is organized into distinct white containers against a soft off-white background to reduce visual fatigue during long analytical sessions. The emotional response is one of stability, transparency, and logical rigor.

Key principles:
- **Data Integrity:** Financial figures are never obscured by decorative elements.
- **Calibrated Hierarchy:** Color is used functionally to indicate risk levels, not for decoration.
- **Predictability:** Consistent component behavior across complex data entry and visualization.

## Colors
The palette is built on a foundation of "Trust Blue" tones and high-utility status colors. 

- **Primary & Secondary:** Deep Navy (#0F172A) is used for headers, text, and primary navigation to establish authority. Indigo (#4F46E5) serves as the primary action color for buttons, active states, and links.
- **Semantic Palette:** Defined strictly for credit scoring outcomes. Emerald Green indicates "Safe/Excellent" scores; Amber indicates "Caution/Fair"; Rose Red indicates "High Risk/Poor."
- **Neutrals:** Slate Gray is used for secondary text and icons, ensuring WCAG AA contrast ratios against the white surface backgrounds.
- **Surface:** The background uses a cool off-white (#F8FAFC) to differentiate from the white (#FFFFFF) component cards, creating a subtle sense of depth without relying on heavy shadows.

## Typography
This design system employs a dual-font strategy. **Be Vietnam Pro** is used for headlines to provide a modern, localized character that handles Vietnamese diacritics with exceptional elegance. **Inter** is used for body text and data due to its high legibility and robust support for OpenType features.

**Critical Instruction:** All financial values, credit scores, and table data must use Inter with `font-variant-numeric: tabular-nums`. This ensures that decimals and digits align vertically, allowing users to scan and compare financial figures accurately.

For mobile responsiveness:
- `display-lg` scales down to 36px on mobile devices.
- `headline-lg` scales down to 24px on mobile devices.
- Maintain base body size (16px) across all devices for readability.

## Layout & Spacing
The layout follows a 12-column fluid grid system for desktop, transitioning to 8 columns for tablets and 4 columns for mobile. 

- **Grid Logic:** Use 24px gutters to provide significant breathing room between complex data cards. 
- **Margins:** Page margins are set to 32px on desktop and 16px on mobile.
- **Rhythm:** An 8px linear scale is used for all internal component spacing (padding/margins). 
- **Multi-column Forms:** Complex financial applications should utilize a two-column layout on desktop to minimize eye travel, grouping related fields into titled sections.
- **Dashboard Consistency:** Main dashboard widgets should span 3, 4, 6, or 12 columns to maintain a structured visual rhythm.

## Elevation & Depth
Depth is conveyed through a combination of **Tonal Layering** and **Soft Shadows**. 

1. **Background (Level 0):** #F8FAFC (Off-white).
2. **Card Surface (Level 1):** #FFFFFF (Pure White). 
   - Border: 1px Solid #E2E8F0.
   - Shadow: `0px 4px 6px -1px rgba(15, 23, 42, 0.05), 0px 2px 4px -2px rgba(15, 23, 42, 0.05)`.
3. **Interactive/Overlay (Level 2):** Modals and dropdowns.
   - Border: 1px Solid #E2E8F0.
   - Shadow: `0px 20px 25px -5px rgba(15, 23, 42, 0.1), 0px 8px 10px -6px rgba(15, 23, 42, 0.1)`.

Avoid any use of glassmorphism or background blurs, as they detract from the professional, data-centric focus of the system.

## Shapes
The design system uses a generous roundedness profile to soften the technical nature of credit data.

- **Standard Cards:** 20px (`rounded-2xl`) corner radius.
- **Buttons & Inputs:** 8px (`rounded-lg`) corner radius to maintain a professional "tool-like" feel.
- **Status Badges/Pills:** Fully rounded (pill-shaped) to distinguish them from interactive buttons.

This geometric contrast helps users instantly differentiate between "Information Containers" (Soft 20px) and "Actionable Elements" (Precise 8px).

## Components

### Buttons
- **Primary:** Solid Indigo (#4F46E5) with white text. High emphasis.
- **Secondary:** White background with 1px border (#E2E8F0) and Navy text.
- **Tertiary/Ghost:** No background or border, Indigo text. Used for less frequent actions.

### Cards & Gauges
- **Credit Score Gauge:** Use a semi-circular stroke with a width of 12px. The stroke color dynamically updates based on the score (Green/Amber/Red). The score value is centered in `display-lg` weight.
- **SHAP/Feature Importance Bars:** Horizontal bar charts with 4px rounded ends. Positive impact features use Emerald Green; negative impact features use Rose Red.

### Input Fields
- **Default State:** 1px border (#E2E8F0) with a 16px horizontal padding.
- **Focus State:** 2px border (#4F46E5) with a soft Indigo outer glow (4px spread, 10% opacity).
- **Validation:** Clear error messages in Rose Red (#F43F5E) positioned directly below the field.

### Status Indicators
- Use a combination of a colored "Dot" icon and text (e.g., "• High Risk"). This ensures accessibility for color-blind users who may struggle with color-only indicators.

### Lists & Data Tables
- Header rows use `label-caps` typography with a subtle background tint (#F1F5F9).
- Row separators are 1px #F1F5F9. Row hover state uses #F8FAFC.