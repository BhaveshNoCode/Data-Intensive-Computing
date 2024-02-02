import { Injectable } from '@angular/core';

import { HttpClient, HttpHeaders, HttpEvent, HttpRequest } from '@angular/common/http';
import { Observable } from '../../../node_modules/rxjs';
import { map } from 'rxjs/operators';
import { DomSanitizer, SafeResourceUrl} from '@angular/platform-browser';

@Injectable({
  providedIn: 'root'
})
export class PredictService {

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) { }

  predict(form_obj: any[]): Observable<any>{
    const httpOptions = {
      headers: new HttpHeaders({ 'Content-Type': 'application/json' })
    };
    var to_send = {
      info: form_obj
    };
    console.log(form_obj);
    return this.http.post<any>("predict", to_send, httpOptions);
  }

  getPie(): Observable<SafeResourceUrl>{
    return this.http.get("/pie", { responseType: 'blob' }).pipe(map(x => {
      const urlToBlob = window.URL.createObjectURL(x)
      return this.sanitizer.bypassSecurityTrustResourceUrl(urlToBlob);
    }));
  }

  getBar(): Observable<SafeResourceUrl>{
    return this.http.get("/bar", { responseType: 'blob' }).pipe(map(x => {
      const urlToBlob = window.URL.createObjectURL(x)
      return this.sanitizer.bypassSecurityTrustResourceUrl(urlToBlob);
    }));
  }

  test2(): Observable<any> {
    return this.http.get("/test2");
  }

  upload(file: File): Observable<any> {
    const httpOptions = {
      headers: new HttpHeaders({ 'Content-Type': 'multipart/form-data' })
    };
    var formData = new FormData();
    formData.append('file', file);
    return this.http.post<any>("upload", formData);
  }

  // upload(file: File): Observable<HttpEvent<any>> {
  //   // const formData: FormData = new FormData();

  //   // formData.append('file', file);

  //   const req = new HttpRequest('POST', `/upload`, file, {
  //     reportProgress: true,
  //     responseType: 'json'
  //   });

  //   return this.http.request(req);
  // }

}
