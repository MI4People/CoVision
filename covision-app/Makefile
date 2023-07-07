.PHONY: default
default: android-release-unsigned

android-release-unsigned:
	ionic build
	ionic capacitor copy android
	(cd android && ./gradlew assembleDebug)
